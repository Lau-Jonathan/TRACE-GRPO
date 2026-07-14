"""Build BEACON-aligned ScienceWorld parquets for TRACE-GRPO.

Protocol (the paper §4.3/§4.4):
1. Train rows are placeholders; runtime always samples task/variation from
   BEACON L0 train pool.
2. Validation rows are fixed (task, variation) entries from BEACON L0 test
   pool using stratified sampling over task ids.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from datasets import Dataset


def _default_beacon_idx_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return (
        repo_root
        / "BEACON"
        / "agent_system"
        / "environments"
        / "env_package"
        / "sciworld"
        / "variations_idx"
        / "L0_idx.json"
    )


def _load_beacon_idx(path: Path) -> dict[str, list[tuple[int, int]]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict JSON at {path}, got {type(payload).__name__}")
    out: dict[str, list[tuple[int, int]]] = {}
    for split, rows in payload.items():
        out[str(split)] = [(int(t), int(v)) for t, v in rows]
    return out


def _load_task_names() -> list[str]:
    from scienceworld import ScienceWorldEnv  # noqa: WPS433

    env = ScienceWorldEnv()
    try:
        return list(env.get_task_names())
    finally:
        try:
            env.close()
        except Exception:
            pass


def _round_robin_merge(groups: dict[int, list[tuple[int, int]]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    task_ids = sorted(groups)
    cursor = 0
    while True:
        active = False
        for tid in task_ids:
            seq = groups[tid]
            if cursor < len(seq):
                out.append(seq[cursor])
                active = True
        if not active:
            break
        cursor += 1
    return out


def _stratified_val_pairs(pool: Iterable[tuple[int, int]], n_val: int) -> list[tuple[int, int]]:
    by_task: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for task_id, variation in pool:
        by_task[int(task_id)].append((int(task_id), int(variation)))
    if not by_task:
        return []

    task_ids = sorted(by_task)
    n_tasks = len(task_ids)
    base = max(n_val // n_tasks, 0)

    selected_by_task: dict[int, list[tuple[int, int]]] = {tid: [] for tid in task_ids}
    remaining_by_task: dict[int, list[tuple[int, int]]] = {}
    for tid in task_ids:
        seq = by_task[tid]
        take = min(base, len(seq))
        selected_by_task[tid] = seq[:take]
        remaining_by_task[tid] = seq[take:]

    selected_cnt = sum(len(v) for v in selected_by_task.values())
    leftover = max(n_val - selected_cnt, 0)
    if leftover > 0:
        candidates = sorted(
            task_ids,
            key=lambda tid: (-len(remaining_by_task[tid]), tid),
        )
        idx = 0
        while leftover > 0 and candidates:
            tid = candidates[idx % len(candidates)]
            if remaining_by_task[tid]:
                selected_by_task[tid].append(remaining_by_task[tid].pop(0))
                leftover -= 1
            candidates = [c for c in candidates if remaining_by_task[c]]
            idx += 1

    return _round_robin_merge(selected_by_task)


def _train_rows(size: int) -> list[dict]:
    rows = []
    for idx in range(int(size)):
        rows.append(
            {
                "data_source": "sciworld",
                "prompt": [{"role": "user", "content": ""}],
                "ability": "agent",
                "agent_name": "trace_sciworld",
                "split": "train",
                "extra_info": {"split": "train", "index": idx},
            }
        )
    return rows


def _val_rows(
    pairs: list[tuple[int, int]],
    task_names: list[str],
) -> list[dict]:
    rows = []
    for idx, (task_id, variation) in enumerate(pairs):
        if task_id < 0 or task_id >= len(task_names):
            raise IndexError(f"task_id={task_id} out of range [0, {len(task_names)})")
        rows.append(
            {
                "data_source": "sciworld",
                "prompt": [{"role": "user", "content": ""}],
                "ability": "agent",
                "agent_name": "trace_sciworld",
                "split": "val",
                "sciworld_task": task_names[task_id],
                "sciworld_var": int(variation),
                "extra_info": {
                    "split": "val",
                    "index": idx,
                    "task_id": int(task_id),
                    "variation": int(variation),
                },
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/sciworld_beacon")
    parser.add_argument("--train_data_size", type=int, default=16)
    parser.add_argument("--val_data_size", type=int, default=128)
    parser.add_argument("--beacon_idx", default=str(_default_beacon_idx_path()))
    args = parser.parse_args()

    local_dir = Path(args.local_dir).expanduser()
    local_dir.mkdir(parents=True, exist_ok=True)

    beacon_idx = Path(args.beacon_idx).expanduser()
    pools = _load_beacon_idx(beacon_idx)
    if "test" not in pools:
        raise KeyError(f"BEACON index missing 'test' split: {beacon_idx}")
    task_names = _load_task_names()
    val_pairs = _stratified_val_pairs(pools["test"], int(args.val_data_size))

    train = Dataset.from_list(_train_rows(int(args.train_data_size)))
    val = Dataset.from_list(_val_rows(val_pairs, task_names))
    train.to_parquet(str(local_dir / "train.parquet"))
    val.to_parquet(str(local_dir / "val.parquet"))

    print(f"beacon_idx={beacon_idx}")
    print(f"wrote {len(train)} train rows to {local_dir / 'train.parquet'}")
    print(
        f"wrote {len(val)} val rows to {local_dir / 'val.parquet'} "
        f"(requested={int(args.val_data_size)})"
    )


if __name__ == "__main__":
    main()
