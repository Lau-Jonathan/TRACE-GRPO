"""Smoke test for the actor-worker TRACE-GRPO L3 critique_delta RPC."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from verl import DataProto
from verl.workers.fsdp_workers import ActorRolloutRefWorker

from trace_grpo.patches.capa_forward import ReferenceCritiqueForward
from trace_grpo.patches.critique_conditioned_provider import (
    CritiqueConditionedProvider,
    TrajectoryAnnotation,
    TurnAnnotation,
)


def _make_tiny_gpt2_cuda(seed: int = 0):
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    cfg = GPT2Config(
        vocab_size=64,
        n_positions=128,
        n_embd=32,
        n_layer=2,
        n_head=4,
        attn_implementation="eager",
    )
    model = GPT2LMHeadModel(cfg).eval().cuda()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for actor-worker smoke")
def test_fsdp_worker_l3_rpc_matches_reference_provider():
    model = _make_tiny_gpt2_cuda()
    prompt_length = 4
    response_length = 20
    traj_ids = torch.randint(1, 64, (prompt_length + response_length,), dtype=torch.long).tolist()
    turn = TurnAnnotation(
        history_index=0,
        response_start=3,
        response_end=9,
        critique_token_ids=[10, 11, 12],
        q_t=-0.7,
    )
    annotation = TrajectoryAnnotation(
        traj_token_ids=traj_ids,
        prompt_length=prompt_length,
        response_length=response_length,
        turns=[turn],
        traj_offset_in_batch=0,
    )

    full = torch.tensor(traj_ids, dtype=torch.long, device="cuda").unsqueeze(0)
    with torch.no_grad():
        baseline_logits = model(full, use_cache=False).logits.float()
    targets = full[0, prompt_length:]
    score_logits = baseline_logits[0, prompt_length - 1 : -1, :]
    old_lp = F.log_softmax(score_logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1).cpu()

    provider = CritiqueConditionedProvider(
        forward_fn=ReferenceCritiqueForward(model=model, device="cuda", dtype=torch.float32)
    )
    expected = provider.compute_critique_delta(
        trajectories=[annotation],
        old_logprobs=old_lp.unsqueeze(0),
    )

    worker_self = SimpleNamespace(
        _is_actor=True,
        _is_offload_param=False,
        actor=SimpleNamespace(actor_module=model, param_dtype=torch.float32),
        world_size=1,
    )
    data = DataProto.from_dict(
        tensors={
            "old_log_probs": old_lp.unsqueeze(0),
            "input_ids": full.cpu(),
        },
        non_tensors={"trace_trajectories": np.array([annotation], dtype=object)},
    )
    out = ActorRolloutRefWorker.compute_trace_critique_delta(worker_self, data)

    assert "critique_delta" in out.batch
    assert torch.allclose(out.batch["critique_delta"], expected, atol=1e-6)
    assert out.batch["critique_delta"][0, turn.response_start : turn.response_end].abs().sum() > 0
