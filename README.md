# TRACE-GRPO — anonymous submission code release

Anonymized reference implementation for **TRACE-GRPO** (three-level
credit-assignment GRPO with Critique-Aware Paged Attention, CAPA).
Companion code for the corresponding paper submission; author identity
and institutional affiliation are intentionally removed.

## Contents

```
.
├── __init__.py, pyproject.toml
├── configs/
│   ├── agent_loops.yaml                       # trainer agent-loop wiring
│   └── algorithm/trace_grpo.yaml                 # L1/L2/L3 hyperparameters + judge cfg
├── patches/                                   # verl advantage-estimator and forward hooks
│   ├── level3_patch.py                        # L1 + L2 + L3 advantage estimator (registered adv_estimator)
│   ├── capa.py, capa_forward.py               # CAPA paged attention + wrapper
│   ├── critique_conditioned_provider.py       # provider that ties CAPA into the trainer step
│   ├── conditioned_forward.py                 # delta-provider registry
│   ├── counterfactual_provider.py             # optional counterfactual-mask teacher variant
│   ├── context.py                             # shared L3 stats stash
│   ├── trainer_hook.py                        # trainer-side integration hook
│   └── sciworld_reward_manager.py             # BEACON-aligned reward manager
├── self_supervised/                           # non-LLM teachers
│   ├── env_score_delta_annotator.py           # q_t = sign(Δenv_score), no API calls
│   └── counterfactual_mask_annotator.py       # optional counterfactual variant
├── workers/reward_manager/text_feedback/      # LLM-judge teacher pipeline
│   ├── manager.py                             # verl RewardManager wiring
│   ├── _annotator.py                          # async OpenAI-compatible client
│   └── prompt.py                              # judge prompt template + JSON schema
├── agent_loops/                               # SciWorld / AlfWorld agent loops (verl AgentLoopBase)
├── launchers/main_ppo.py                      # thin trainer entry point
├── scripts/                                   # launch scripts + data build
│   ├── run_sciworld_trace_l3_*.sh           # main SciWorld TRACE-GRPO launches
│   ├── run_alfworld_trace_l3_*.sh           # main AlfWorld TRACE-GRPO launches
│   ├── run_*_vanilla_grpo_*.sh                # matching GRPO baselines
│   ├── ablation_l1l2_no_l3_8xh100.sh          # L1+L2 (no L3) ablation
│   ├── run_capa_validation_suite.sh           # one-click CAPA correctness suite
│   ├── build_sciworld_parquet_beacon.py       # BEACON dataset construction, SciWorld
│   └── build_alfworld_parquet_beacon.py       # BEACON dataset construction, AlfWorld
├── tests/                                     # unit tests (see §Tests below)
└── utils/beacon_metrics.py                    # BEACON metric helpers
```

Design and file-by-file rationale is written up in the paper's appendices;
this repository provides the runnable implementation.

## Environment

TRACE-GRPO is built on top of an unmodified [verl](https://github.com/volcengine/verl)
RL training framework. Only additive changes are made through verl's
plug-in registries (`register_adv_est`, `AgentLoopBase`,
`ALL_ATTENTION_FUNCTIONS`); no verl source files are patched by this
repository.

Minimum versions:

- Python ≥ 3.10
- PyTorch ≥ 2.5 (CUDA build for training; CPU is fine for tests)
- `transformers` ≥ 4.51 (≥ 4.57 for the packed CAPA attention dispatch)
- `verl` (main branch; the trainer entry point in `launchers/main_ppo.py`
  is a shim over `verl.trainer.main_ppo`)
- `flash-attn` ≥ 2.7 (only needed for the FA2 bf16 GPU test; CPU tests
  run without it)
- `scienceworld` for the SciWorld environment / `alfworld` for AlfWorld

Editable install:

```bash
git clone <this repo> trace_grpo && cd trace_grpo
pip install -e .              # core deps
pip install -e .[flashattn]   # optional: FA2 for the GPU numerical test
pip install -e .[sciworld]    # SciWorld environment
pip install -e .[test]        # pytest
# install verl from source per its own README
```

## Data preparation

`build_sciworld_parquet_beacon.py` and `build_alfworld_parquet_beacon.py`
implement the BEACON-aligned stratified evaluation-set construction
described in the paper's Appendix. Both write `train.parquet` /
`val.parquet` to a directory of your choice:

```bash
DATA_DIR=data/sciworld_beacon \
python scripts/build_sciworld_parquet_beacon.py --out $DATA_DIR

DATA_DIR=data/alfworld_beacon \
python scripts/build_alfworld_parquet_beacon.py --out $DATA_DIR
```

The BEACON pool JSON (`L0_idx.json`) is loaded from an environment
variable (`BEACON_SCIWORLD_POOL` / `BEACON_ALFWORLD_POOL`) or
`--pool <path>`. Neither pool is redistributed here.

## Training

The launch scripts under `scripts/` expose the full training pipeline.
Each script reads a small set of env vars (data path, backbone
checkpoint, teacher endpoint), then invokes `python -m
verl.trainer.main_ppo` with all hyperparameters. Sample:

```bash
# TRACE-GRPO with an env-score self-supervised teacher (no API calls):
export TRACE_GRPO_ROOT=$PWD
export CKPT_DIR=/path/to/Qwen2.5-7B-Instruct
export DATA_DIR=data/sciworld_beacon
TRACE_GRPO_TEACHER_KIND=env_score \
bash scripts/run_sciworld_trace_l3_beacon_prod_llmjudge_7b.sh 200

# Vanilla GRPO baseline (same trainer, no teacher, no L3):
bash scripts/run_sciworld_vanilla_grpo_beacon_prod_llmjudge_7b.sh 200

# L1 + L2 ablation (L3 disabled at algorithm level):
bash scripts/ablation_l1l2_no_l3_8xh100.sh

# With an LLM-judge teacher, set the endpoint:
export TEXT_FEEDBACK_BASE_URL=https://your-endpoint.example.com/v1/chat/completions
export TEXT_FEEDBACK_MODEL=your-judge-model
export TEXT_FEEDBACK_API_KEY=sk-...
TRACE_GRPO_TEACHER_KIND=llm \
bash scripts/run_sciworld_trace_l3_beacon_prod_llmjudge_7b.sh 200
```

Each launcher writes a fresh output directory under `outputs/`; consecutive
runs never share directories (the launchers refuse to reuse a non-empty
run dir unless `ALLOW_REUSE_RUN_DIR=1`).

## Evaluation

Evaluation is done by the same trainer via periodic validation
(`trainer.test_freq` in the launcher) and can be reproduced offline
from the validation JSONL files that every run drops into
`outputs/<run>/validation_jsonl/*.jsonl`. Each JSONL row records
per-task success, score, episode length, and category, which is what
the main results table and difficulty-strata figure aggregate.

## CAPA correctness test

The one-click validation suite in `scripts/run_capa_validation_suite.sh`
runs the CPU-only equivalence tests unconditionally and, in `MODE=full`,
adds the bf16 + FlashAttention-2 GPU drift check:

```bash
# CPU only, ~5s, no CUDA required:
MODE=quick bash scripts/run_capa_validation_suite.sh

# Adds bf16 + FA2 GPU test on device 0:
MODE=full  bash scripts/run_capa_validation_suite.sh
```

Expected outcome (matches the reference hardware in the paper):

- `capa_prefix_kv_equivalence`      — 6/6 pass, fp32 bit-exact
- `capa_prefix_kv_equivalence_gqa`  — 1/1 pass (GQA layout)
- `capa_qwen2_packed_vs_reference`  — 1/1 pass (packed vs. per-turn)
- `capa_fa2_bf16_drift`             — 1/1 pass, max |Δ| < 0.1

## Tests

```bash
pytest -q tests/                                          # full suite
pytest -q tests/test_advantage_l1l2l3.py                  # L1+L2+L3 golden example
pytest -q tests/test_prefix_kv_equivalence.py             # CAPA CPU/fp32 bit-exact equivalence
pytest -q tests/test_capa_forward.py                      # packed CAPA vs. per-turn reference
pytest -q tests/test_critique_provider.py                 # provider scheduling of annotated turns
pytest -q tests/test_reward_and_teacher.py                # reward manager + teacher plumbing
pytest -q tests/test_actor_l3_rpc_smoke.py                # FSDP actor L3 RPC integration smoke
pytest -q tests/test_beacon_score_metrics.py              # BEACON metric helpers
pytest -q tests/test_invalid_action_penalty.py            # BEACON invalid-action penalty
pytest -q tests/test_sciworld_runner.py                   # SciWorld agent loop
pytest -q tests/test_alfworld_runner.py                   # AlfWorld agent loop
pytest -q tests/test_trainer_hook.py                      # trainer-side hook
```

The `test_llm_judge.py` file contains a live-endpoint smoke test that is
skipped by default. Set `TRACE_GRPO_RUN_LIVE_TESTS=1` and
`TEXT_FEEDBACK_API_KEY=<token>` to hit your configured judge endpoint.

## What is not included

- **Model weights.** No checkpoints are redistributed. All experiments
  start from a publicly available instruction-tuned backbone (Qwen2.5-7B
  or Qwen2.5-1.5B); path is passed via `CKPT_DIR`.
- **Environments.** ScienceWorld / AlfWorld are third-party packages and
  are installed as dependencies rather than vendored.
- **Training logs and W&B history.** Available on request through the
  paper's supplementary materials; kept out of the anonymized code
  release so that no runtime identifiers leak in.

## Minimal smoke test

Once the environment is set up:

```bash
# 1. CAPA CPU equivalence (no CUDA, no data, no model download):
MODE=quick bash scripts/run_capa_validation_suite.sh

# 2. L1+L2+L3 golden example from the paper's Appendix:
pytest -q tests/test_advantage_l1l2l3.py

# 3. Trainer-side hook + reward manager + teacher plumbing:
pytest -q tests/test_trainer_hook.py tests/test_reward_and_teacher.py \
          tests/test_critique_provider.py
```

Expected output: all of the above pass in under 30 seconds on a laptop CPU.
