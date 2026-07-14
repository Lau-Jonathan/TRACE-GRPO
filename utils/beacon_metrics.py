from __future__ import annotations

from typing import Any

import numpy as np


def beacon_row_weighted_scores_by_source(
    data_sources: Any,
    scores: Any,
    num_turns: Any,
) -> dict[str, float]:
    """Match BEACON's row-expanded ScienceWorld test_score aggregation.

    BEACON stores one validation row per active environment step and assigns
    the full episode reward to every row. Packed trajectories store one row per
    episode, so the equivalent mean is episode_reward weighted by num_turns.
    """

    data_arr = np.asarray(data_sources)
    score_arr = np.asarray(scores, dtype=np.float64)
    turn_arr = np.asarray(num_turns, dtype=np.float64)
    if not (data_arr.shape[0] == score_arr.shape[0] == turn_arr.shape[0]):
        raise ValueError(
            "data_sources, scores, and num_turns must have the same leading length"
        )

    out: dict[str, float] = {}
    for data_source in np.unique(data_arr):
        mask = data_arr == data_source
        if not np.any(mask):
            continue
        weights = np.maximum(turn_arr[mask], 1.0)
        out[str(data_source)] = float(np.average(score_arr[mask], weights=weights))
    return out


def beacon_success_rate_batch_means(
    data_source_batches: Any,
    won_batches: Any,
) -> dict[str, float]:
    """Match BEACON's validation success_rate aggregation.

    BEACON's rollout loop first computes success_rate inside each validation
    rollout batch, stores that batch-level scalar on every row, then the trainer
    averages those batch-level scalars. This intentionally preserves that
    aggregation instead of converting it into a sample-weighted mean.
    """

    if len(data_source_batches) != len(won_batches):
        raise ValueError("data_source_batches and won_batches must have the same length")

    key2batch_means: dict[str, list[float]] = {}
    for data_sources, won_values in zip(data_source_batches, won_batches, strict=True):
        data_arr = np.asarray(data_sources)
        won_arr = np.asarray(won_values, dtype=np.float64)
        if data_arr.shape[0] != won_arr.shape[0]:
            raise ValueError("each data_source batch and won batch must have the same length")
        if won_arr.shape[0] == 0:
            continue

        key2batch_means.setdefault("success_rate", []).append(float(np.mean(won_arr)))
        for data_source in np.unique(data_arr):
            mask = data_arr == data_source
            if not np.any(mask):
                continue
            key = f"{data_source}_success_rate"
            key2batch_means.setdefault(str(key), []).append(float(np.mean(won_arr[mask])))

    return {key: float(np.mean(values)) for key, values in key2batch_means.items() if values}
