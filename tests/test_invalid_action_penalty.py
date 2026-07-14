"""BEACON-aligned invalid-action penalty.

Mirrors ``BEACON/verl/trainer/ppo/ray_trainer.py:apply_invalid_action_penalty``
(charges 0.5 per invalid env step on the row's last response token), but
applied to packed trajectory rows where one packed row aggregates the per-step
charges into ``coef * num_invalid_turns`` on the trajectory's last response
token. The two formulations sum to the same per-trajectory reduction in
``token_level_scores``, which is what GRPO advantages observe.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl import DataProto


def _load_apply_invalid_action_penalty():
    """Load just the standalone function from ray_trainer.py.

    ``verl.trainer.ppo.ray_trainer`` triggers vLLM / CUDA initialization at
    import time, which fails in CPU-only test environments. Pull the function
    out via importlib so the unit test stays hermetic.
    """
    src_path = (
        Path(__file__).resolve().parents[2] / "verl" / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    )
    src = src_path.read_text(encoding="utf-8")
    start = src.index("def apply_invalid_action_penalty(")
    end = src.index("def apply_kl_penalty(", start)
    snippet = "import numpy as np\nimport torch\nfrom verl import DataProto\n\n" + src[start:end]
    ns: dict = {}
    exec(snippet, ns)
    return ns["apply_invalid_action_penalty"]


apply_invalid_action_penalty = _load_apply_invalid_action_penalty()


def _make_data(*, scores: torch.Tensor, valid_lens: list[int], num_invalid_turns: list[int],
               num_turns: list[int] | None = None) -> DataProto:
    bsz, response_len = scores.shape
    prompt_len = 2
    prompts = torch.zeros(bsz, prompt_len, dtype=torch.long)
    response_attn = torch.zeros(bsz, response_len, dtype=torch.long)
    for i, vl in enumerate(valid_lens):
        response_attn[i, :vl] = 1
    attention_mask = torch.cat([torch.ones(bsz, prompt_len, dtype=torch.long), response_attn], dim=-1)
    batch = TensorDict(
        {
            "prompts": prompts,
            "attention_mask": attention_mask,
            "token_level_scores": scores.clone(),
        },
        batch_size=bsz,
    )
    non_tensor = {
        "num_invalid_turns": np.asarray(num_invalid_turns, dtype=np.int64),
    }
    if num_turns is not None:
        non_tensor["__num_turns__"] = np.asarray(num_turns, dtype=np.int32)
    return DataProto(batch=batch, non_tensor_batch=non_tensor)


def test_invalid_action_penalty_subtracts_at_last_valid_response_token():
    bsz, response_len = 2, 6
    base_scores = torch.zeros(bsz, response_len)
    base_scores[0, 3] = 12.0  # arbitrary BEACON score sitting at last response token
    base_scores[1, 5] = 2.0

    data = _make_data(
        scores=base_scores,
        valid_lens=[4, 6],
        num_invalid_turns=[2, 0],
        num_turns=[5, 4],
    )
    data, metrics = apply_invalid_action_penalty(data, invalid_action_penalty_coef=0.5)

    out = data.batch["token_level_scores"]
    assert pytest.approx(out[0, 3].item()) == 12.0 - 0.5 * 2  # 11.0
    assert pytest.approx(out[1, 5].item()) == 2.0
    # valid_action_ratio = 1 - sum(invalid) / sum(num_turns) = 1 - 2/9
    assert pytest.approx(metrics["episode/valid_action_ratio"]) == 1.0 - 2.0 / 9.0


def test_invalid_action_penalty_no_invalid_is_noop():
    bsz, response_len = 2, 4
    scores = torch.tensor([[0, 0, 0, 5.0], [0, 0, 7.0, 0]])
    data = _make_data(
        scores=scores,
        valid_lens=[4, 3],
        num_invalid_turns=[0, 0],
        num_turns=[3, 3],
    )
    data, metrics = apply_invalid_action_penalty(data, invalid_action_penalty_coef=0.5)
    torch.testing.assert_close(data.batch["token_level_scores"], scores)
    assert metrics["episode/valid_action_ratio"] == 1.0


def test_invalid_action_penalty_falls_back_when_num_turns_absent():
    bsz, response_len = 1, 3
    scores = torch.tensor([[0.0, 0.0, 1.0]])
    data = _make_data(
        scores=scores,
        valid_lens=[3],
        num_invalid_turns=[1],
    )
    data, metrics = apply_invalid_action_penalty(data, invalid_action_penalty_coef=0.5)
    assert pytest.approx(data.batch["token_level_scores"][0, 2].item()) == 0.5
    # Without __num_turns__ we still emit a metric (fraction of trajectories with no invalid turns).
    assert "episode/valid_action_ratio" in metrics
