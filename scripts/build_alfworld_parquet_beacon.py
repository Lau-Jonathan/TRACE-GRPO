"""Build BEACON-aligned AlfWorld parquets + cached pool index for TRACE-GRPO.

Protocol (mirrors :mod:`build_sciworld_parquet_beacon`):
  1. Train rows are placeholders; runtime always samples ``alfworld_game_index``
     from the cached pool.
  2. Validation rows are fixed ``alfworld_game_index`` entries drawn from the
     ``valid_seen`` (or ``valid_unseen``) split using stratified sampling
     over alfworld task types so the val mix is comparable to BEACON.

Pool file (``data/alfworld_beacon_prod/index.json``):

    {
      "train":                   [int, ...],
      "valid_seen":              [int, ...],
      "valid_unseen":            [int, ...],
      "eval_in_distribution":    [int, ...],   # alias of valid_seen
      "eval_out_of_distribution":[int, ...],   # alias of valid_unseen
    }

The values are integer indices into the ordered ``AlfredTWEnv.game_files``
list for each split. The runtime
:class:`trace_grpo.agent_loops.alfworld_agent_loop.AlfWorldAgentLoop`
re-instantiates ``AlfredTWEnv`` with the same config and treats the int as
an index into ``game_files`` (modulo length), so this pool stays stable as
long as ``$ALFWORLD_DATA`` doesn't change.

A companion ``meta.json`` records ``task_type_by_index`` for each split so
the stratified sampler in this script can do its job without re-walking
the filesystem on every invocation.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from datasets import Dataset


_TASK_TYPES = [
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
]

_SPLITS = [
    ("train", "data_path", "train"),
    ("valid_seen", "eval_id_data_path", "eval_in_distribution"),
    ("valid_unseen", "eval_ood_data_path", "eval_out_of_distribution"),
]


def _default_config_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return (
        repo_root
        / "BEACON"
        / "agent_system"
        / "environments"
        / "env_package"
        / "alfworld"
        / "configs"
        / "config_tw.yaml"
    )


def _classify_task_type(game_file: str) -> str:
    """Recover task_type from the adjacent ``traj_data.json``.

    AlfredTWEnv already filters by task_type on init; we re-parse here to
    drive the stratified sampler. Falls back to ``"unknown"`` on any IO
    failure so we don't crash a full build over one missing file.
    """
    try:
        traj_path = os.path.join(os.path.dirname(game_file), "traj_data.json")
        with open(traj_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return str(data.get("task_type", "unknown"))
    except Exception:
        return "unknown"


def _walk_split(config_path: Path, train_eval: str) -> list[str]:
    """Return the canonical game_files ordering AlfredTWEnv would produce."""
    import yaml  # noqa: WPS433

    # Lazy import: pulls in textworld and alfworld native libs.
    from agent_system.environments.env_package.alfworld.alfworld.agents.environment.alfred_tw_env import (  # noqa: WPS433
        AlfredTWEnv,
    )

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    env = AlfredTWEnv(config, train_eval=train_eval)
    return list(env.game_files)


def _stratified_val_indices(
    games: list[str],
    task_types: list[str],
    n_val: int,
) -> list[int]:
    """Pick ``n_val`` indices spread across the 6 alfworld task types."""
    by_type: dict[str, list[int]] = defaultdict(list)
    for idx, tt in enumerate(task_types):
        by_type[tt].append(idx)

    type_keys = [tt for tt in _TASK_TYPES if tt in by_type]
    if not type_keys:
        return list(range(min(n_val, len(games))))

    base = max(n_val // len(type_keys), 0)
    selected: dict[str, list[int]] = {tt: by_type[tt][:base] for tt in type_keys}
    remaining: dict[str, list[int]] = {tt: by_type[tt][base:] for tt in type_keys}

    leftover = max(n_val - sum(len(v) for v in selected.values()), 0)
    while leftover > 0:
        progress = False
        for tt in type_keys:
            if remaining[tt]:
                selected[tt].append(remaining[tt].pop(0))
                leftover -= 1
                progress = True
                if leftover == 0:
                    break
        if not progress:
            break

    out: list[int] = []
    cursor = 0
    while True:
        active = False
        for tt in type_keys:
            if cursor < len(selected[tt]):
                out.append(selected[tt][cursor])
                active = True
        if not active:
            break
        cursor += 1
    return out


def _train_rows(size: int) -> list[dict]:
    rows = []
    for idx in range(int(size)):
        rows.append(
            {
                "data_source": "alfworld",
                "prompt": [{"role": "user", "content": ""}],
                "ability": "agent",
                "agent_name": "trace_alfworld",
                "split": "train",
                "extra_info": {"split": "train", "index": idx},
            }
        )
    return rows


def _val_rows(
    indices: list[int],
    *,
    alfworld_split: str,
    task_types: list[str],
) -> list[dict]:
    rows = []
    for row_idx, game_idx in enumerate(indices):
        rows.append(
            {
                "data_source": "alfworld",
                "prompt": [{"role": "user", "content": ""}],
                "ability": "agent",
                "agent_name": "trace_alfworld",
                "split": "val",
                "alfworld_split": alfworld_split,
                "alfworld_game_index": int(game_idx),
                "extra_info": {
                    "split": "val",
                    "index": row_idx,
                    "alfworld_split": alfworld_split,
                    "alfworld_game_index": int(game_idx),
                    "task_type": task_types[game_idx] if game_idx < len(task_types) else "unknown",
                },
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/alfworld_beacon_prod")
    parser.add_argument("--train_data_size", type=int, default=16)
    parser.add_argument("--val_data_size", type=int, default=128)
    parser.add_argument(
        "--val_split",
        default="valid_seen",
        choices=["valid_seen", "valid_unseen"],
        help="Which alfworld split to draw fixed val rows from.",
    )
    parser.add_argument(
        "--config_path",
        default=str(_default_config_path()),
        help="Path to alfworld config_tw.yaml.",
    )
    parser.add_argument(
        "--rebuild_index",
        action="store_true",
        help="Re-walk $ALFWORLD_DATA even if index.json already exists.",
    )
    args = parser.parse_args()

    local_dir = Path(args.local_dir).expanduser()
    local_dir.mkdir(parents=True, exist_ok=True)
    index_path = local_dir / "index.json"
    meta_path = local_dir / "meta.json"
    config_path = Path(args.config_path).expanduser()

    if args.rebuild_index or not index_path.exists() or not meta_path.exists():
        index: dict[str, list[int]] = {}
        meta: dict[str, dict] = {}
        for our_name, _yaml_key, train_eval in _SPLITS:
            games = _walk_split(config_path, train_eval=train_eval)
            task_types = [_classify_task_type(g) for g in games]
            index[our_name] = list(range(len(games)))
            meta[our_name] = {
                "num_games": len(games),
                "task_type_by_index": task_types,
                "train_eval": train_eval,
            }
            print(f"[alfworld] split={our_name!r}: {len(games)} games")
        # Aliases used by the agent loop.
        index["eval_in_distribution"] = list(index["valid_seen"])
        index["eval_out_of_distribution"] = list(index["valid_unseen"])

        with index_path.open("w", encoding="utf-8") as f:
            json.dump(index, f)
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f)
    else:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"[alfworld] reusing cached index at {index_path}")

    val_meta = meta.get(args.val_split, {})
    val_task_types: list[str] = list(val_meta.get("task_type_by_index", []))
    if not val_task_types:
        raise RuntimeError(
            f"AlfWorld pool for split={args.val_split!r} is empty. "
            f"Check $ALFWORLD_DATA and rerun with --rebuild_index."
        )
    n_val = min(int(args.val_data_size), len(val_task_types))
    val_indices = _stratified_val_indices(
        list(range(len(val_task_types))),
        val_task_types,
        n_val,
    )

    train = Dataset.from_list(_train_rows(int(args.train_data_size)))
    val = Dataset.from_list(
        _val_rows(val_indices, alfworld_split=args.val_split, task_types=val_task_types)
    )
    train.to_parquet(str(local_dir / "train.parquet"))
    val.to_parquet(str(local_dir / "val.parquet"))

    print(f"index={index_path}")
    print(f"wrote {len(train)} train rows to {local_dir / 'train.parquet'}")
    print(
        f"wrote {len(val)} val rows to {local_dir / 'val.parquet'} "
        f"(requested={int(args.val_data_size)}, val_split={args.val_split})"
    )


if __name__ == "__main__":
    main()
