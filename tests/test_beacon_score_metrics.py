from __future__ import annotations

import numpy as np

from trace_grpo.utils.beacon_metrics import (
    beacon_row_weighted_scores_by_source,
    beacon_success_rate_batch_means,
)


def test_sciworld_val_test_score_matches_beacon_row_weighting():
    """BEACON validates one row per active env step.

    Packed trajectories produce one row per episode, so val/sciworld/test_score
    must weight episode rewards by num_turns to match BEACON's row-expanded
    average.
    """

    data_sources = np.asarray(["sciworld", "sciworld"], dtype=object)
    metrics = beacon_row_weighted_scores_by_source(
        data_sources,
        [12.0, 2.0],
        np.asarray([2, 6], dtype=np.int32),
    )

    assert metrics["sciworld"] == (12.0 * 2 + 2.0 * 6) / 8


def test_beacon_row_weighting_groups_by_data_source():
    data_sources = np.asarray(["sciworld", "sciworld", "other"], dtype=object)

    metrics = beacon_row_weighted_scores_by_source(
        data_sources,
        [12.0, 2.0, 5.0],
        [2, 6, 3],
    )

    assert metrics["sciworld"] == (12.0 * 2 + 2.0 * 6) / 8
    assert metrics["other"] == 5.0


def test_success_rate_matches_beacon_batch_mean_aggregation():
    metrics = beacon_success_rate_batch_means(
        [
            np.asarray(["sciworld", "sciworld", "sciworld"], dtype=object),
            np.asarray(["sciworld"], dtype=object),
        ],
        [
            np.asarray([1, 0, 0], dtype=np.int32),
            np.asarray([1], dtype=np.int32),
        ],
    )

    # BEACON averages per-rollout-batch success rates: mean([1/3, 1]).
    assert metrics["success_rate"] == (1.0 / 3.0 + 1.0) / 2.0
    assert metrics["sciworld_success_rate"] == (1.0 / 3.0 + 1.0) / 2.0
