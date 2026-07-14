"""Real-model tests for the critique-conditioned forward.

Uses a tiny GPT-2 instantiated from config (no external download), so
this test runs in a few seconds on CPU and doesn't need GPU/internet.

Key checks:
  - :class:`ReferenceCritiqueForward` returns log-probs that match a
    hand-rolled "score this single conditioned input" computation
    (bit-equal at fp32 eager attention).
  - Plumbed through :class:`CritiqueConditionedProvider`, the resulting
    ``critique_delta`` is non-zero on annotated response tokens and zero
    elsewhere.
  - :class:`CapaCritiqueForward` (currently a delegating shim) produces
    the same numbers — this lets us swap the production class once the
    real CAPA dispatch is wired without breaking the contract.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from trace_grpo.patches.capa import PAGE_SIZE
from trace_grpo.patches.capa_forward import (
    CapaCritiqueForward,
    ReferenceCritiqueForward,
)
from trace_grpo.patches.critique_conditioned_provider import (
    CritiqueConditionedProvider,
    TrajectoryAnnotation,
    TurnAnnotation,
)


def _make_tiny_gpt2(seed: int = 0):
    """Build a randomly-initialised GPT-2 (no download). The model is
    large enough that distinct conditioning meaningfully changes
    log-probs but small enough to forward in milliseconds."""
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    cfg = GPT2Config(
        vocab_size=64,         # tiny tokenizer space
        n_positions=128,
        n_embd=32,
        n_layer=2,
        n_head=4,
        attn_implementation="eager",
    )
    model = GPT2LMHeadModel(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _make_tiny_qwen2(seed: int = 0, *, device: str = "cpu", dtype: torch.dtype = torch.float32):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        attn_implementation="eager",
    )
    model = Qwen2ForCausalLM(cfg).eval().to(device=device, dtype=dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _tiny_traj(seed: int = 1, L: int = 24) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(low=1, high=64, size=(L,), generator=g, dtype=torch.long)


# ---------------------------------------------------------------------------
# Reference forward: bit-equal to manual scoring.
# ---------------------------------------------------------------------------


def _manual_cond_logprobs(
    model,
    traj: torch.Tensor,
    prompt_length: int,
    turn: TurnAnnotation,
) -> torch.Tensor:
    """Score the response tokens of one annotated turn via a clean,
    direct forward — no provider or helper involvement. This is the
    "by-hand" oracle the reference path must match."""
    resp_start_full = prompt_length + turn.response_start
    resp_end_full = prompt_length + turn.response_end
    crit = torch.tensor(turn.critique_token_ids, dtype=torch.long)
    full = torch.cat([traj[:resp_start_full], crit, traj[resp_start_full:resp_end_full]])
    full = full.unsqueeze(0)
    with torch.no_grad():
        logits = model(full).logits.float()
    # Log p(traj[k] | full[:k]) at full position k.
    new_resp_start = resp_start_full + crit.shape[0]
    score_logits = logits[0, new_resp_start - 1 : resp_end_full + crit.shape[0] - 1, :]
    target = full[0, new_resp_start : resp_end_full + crit.shape[0]]
    return F.log_softmax(score_logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)


def test_reference_forward_matches_manual_single_turn():
    model = _make_tiny_gpt2()
    L = 24
    traj = _tiny_traj(L=L)
    prompt_length = 4
    response_length = L - prompt_length

    turn = TurnAnnotation(
        history_index=0,
        response_start=2,
        response_end=8,                 # response-axis frame
        critique_token_ids=[10, 11, 12, 13],
        q_t=-1.0,
    )

    # Manual oracle.
    manual = _manual_cond_logprobs(model, traj, prompt_length, turn)

    # Reference forward.
    ref = ReferenceCritiqueForward(model=model, device="cpu", dtype=torch.float32)
    full = ref(trajectory_tokens=traj, prompt_length=prompt_length, turns=[turn])
    assert full.shape == (response_length,)
    seg = full[turn.response_start : turn.response_end]
    assert torch.allclose(seg, manual, atol=1e-6), f"diff = {(seg - manual).abs().max()}"


def test_reference_forward_zero_outside_annotated_span():
    model = _make_tiny_gpt2()
    traj = _tiny_traj(L=24)
    turn = TurnAnnotation(
        history_index=0,
        response_start=4,
        response_end=8,
        critique_token_ids=[15, 16, 17],
        q_t=+1.0,
    )
    ref = ReferenceCritiqueForward(model=model)
    out = ref(trajectory_tokens=traj, prompt_length=4, turns=[turn])
    # Positions outside the annotated turn must be untouched (zero).
    assert torch.all(out[: turn.response_start] == 0)
    assert torch.all(out[turn.response_end :] == 0)


def test_reference_forward_handles_two_disjoint_turns():
    model = _make_tiny_gpt2()
    L = 32
    traj = _tiny_traj(L=L)
    prompt_length = 4

    turn_a = TurnAnnotation(history_index=0, response_start=0, response_end=4,
                            critique_token_ids=[20, 21], q_t=+1.0)
    turn_b = TurnAnnotation(history_index=2, response_start=10, response_end=15,
                            critique_token_ids=[22, 23, 24], q_t=-1.0)

    ref = ReferenceCritiqueForward(model=model)
    out = ref(trajectory_tokens=traj, prompt_length=prompt_length, turns=[turn_a, turn_b])
    # Both annotated spans should have been written.
    assert out[turn_a.response_start : turn_a.response_end].abs().sum() > 0
    assert out[turn_b.response_start : turn_b.response_end].abs().sum() > 0
    # Gap between them stays zero.
    assert torch.all(out[turn_a.response_end : turn_b.response_start] == 0)
    # And each span numerically matches the manual oracle.
    for turn in (turn_a, turn_b):
        manual = _manual_cond_logprobs(model, traj, prompt_length, turn)
        seg = out[turn.response_start : turn.response_end]
        assert torch.allclose(seg, manual, atol=1e-6)


def test_reference_forward_skips_empty_critique_turns():
    model = _make_tiny_gpt2()
    traj = _tiny_traj(L=24)
    turn = TurnAnnotation(history_index=0, response_start=2, response_end=6,
                          critique_token_ids=[], q_t=0.0)
    ref = ReferenceCritiqueForward(model=model)
    out = ref(trajectory_tokens=traj, prompt_length=4, turns=[turn])
    assert torch.all(out == 0)


# ---------------------------------------------------------------------------
# CapaCritiqueForward (delegates) — same numbers as Reference.
# ---------------------------------------------------------------------------


def test_capa_forward_matches_reference():
    """Until the real packed CAPA forward is wired, CapaCritiqueForward
    delegates. This test pins down the contract so swapping in the real
    implementation later is a strictly numerical change.
    """
    model = _make_tiny_gpt2()
    traj = _tiny_traj(L=28)
    prompt_length = 4
    turn = TurnAnnotation(history_index=0, response_start=4, response_end=10,
                          critique_token_ids=[30, 31, 32, 33], q_t=-1.0)

    ref = ReferenceCritiqueForward(model=model)
    capa = CapaCritiqueForward(model=model)
    out_ref = ref(trajectory_tokens=traj, prompt_length=prompt_length, turns=[turn])
    out_capa = capa(trajectory_tokens=traj, prompt_length=prompt_length, turns=[turn])
    assert torch.allclose(out_ref, out_capa, atol=0)


def test_qwen2_capa_forward_uses_single_packed_path_and_matches_reference():
    model = _make_tiny_qwen2()
    traj = _tiny_traj(L=34)
    prompt_length = 5
    turns = [
        TurnAnnotation(history_index=0, response_start=2, response_end=7,
                       critique_token_ids=[10, 11, 12], q_t=0.7),
        TurnAnnotation(history_index=1, response_start=13, response_end=18,
                       critique_token_ids=[20, 21], q_t=-0.7),
    ]

    ref = ReferenceCritiqueForward(model=model, device="cpu", dtype=torch.float32)
    capa = CapaCritiqueForward(model=model, device="cpu", dtype=torch.float32, page_size=8)
    out_ref = ref(trajectory_tokens=traj, prompt_length=prompt_length, turns=turns)
    out_capa = capa(trajectory_tokens=traj, prompt_length=prompt_length, turns=turns)

    assert capa.last_used_packed is True
    drift = (out_ref - out_capa).abs()
    annotated = torch.zeros_like(drift, dtype=torch.bool)
    for turn in turns:
        annotated[turn.response_start : turn.response_end] = True
    assert drift[annotated].max().item() < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Qwen2 CAPA FA2 smoke")
def test_qwen2_capa_forward_fa2_bf16_matches_reference_with_drift_bound():
    model = _make_tiny_qwen2(device="cuda", dtype=torch.bfloat16)
    traj = _tiny_traj(L=40)
    prompt_length = 5
    turns = [
        TurnAnnotation(history_index=0, response_start=3, response_end=10,
                       critique_token_ids=[10, 11, 12, 13], q_t=1.0),
        TurnAnnotation(history_index=1, response_start=18, response_end=24,
                       critique_token_ids=[20, 21, 22], q_t=-1.0),
    ]

    ref = ReferenceCritiqueForward(model=model, device="cuda", dtype=torch.float32)
    capa = CapaCritiqueForward(
        model=model,
        device="cuda",
        dtype=torch.float32,
        page_size=PAGE_SIZE,
        use_fa2=True,
    )
    out_ref = ref(trajectory_tokens=traj, prompt_length=prompt_length, turns=turns)
    out_capa = capa(trajectory_tokens=traj, prompt_length=prompt_length, turns=turns)

    assert capa.last_used_packed is True
    drift = (out_ref - out_capa).abs()
    annotated = torch.zeros_like(drift, dtype=torch.bool)
    for turn in turns:
        annotated[turn.response_start : turn.response_end] = True
    assert drift[annotated].mean().item() < 0.1


# ---------------------------------------------------------------------------
# End-to-end: provider + reference forward → critique_delta tensor.
# ---------------------------------------------------------------------------


def test_provider_with_real_forward_produces_meaningful_delta():
    model = _make_tiny_gpt2()
    L = 32
    traj_ids = _tiny_traj(L=L).tolist()
    prompt_length = 4
    response_length = L - prompt_length

    turn = TurnAnnotation(history_index=0, response_start=2, response_end=8,
                          critique_token_ids=[40, 41, 42, 43], q_t=-1.0)
    annotation = TrajectoryAnnotation(
        traj_token_ids=traj_ids,
        prompt_length=prompt_length,
        response_length=response_length,
        turns=[turn],
        traj_offset_in_batch=0,
    )

    # Compute the *baseline* old_log_probs ourselves on the same model
    # so cond - old has the right meaning when we subtract.
    full = torch.tensor(traj_ids, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        baseline_logits = model(full).logits.float()
    targets = full[0, prompt_length:]
    score_logits = baseline_logits[0, prompt_length - 1 : -1, :]
    old_lp_row = F.log_softmax(score_logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    old_logprobs = old_lp_row.unsqueeze(0)             # (1, response_length)

    provider = CritiqueConditionedProvider(forward_fn=ReferenceCritiqueForward(model=model))
    delta = provider.compute_critique_delta(
        trajectories=[annotation],
        old_logprobs=old_logprobs,
    )

    # Outside annotated span, delta is zero.
    assert torch.all(delta[0, : turn.response_start] == 0)
    assert torch.all(delta[0, turn.response_end :] == 0)
    # Inside the span, delta is non-zero (different conditioning ⇒ different logits).
    annotated_seg = delta[0, turn.response_start : turn.response_end]
    assert annotated_seg.abs().sum().item() > 1e-4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
