"""Prompt templates + parsers for the LLM judge teacher.

The production path follows the paper v3: score a full trajectory
in one JSON object, compute ``q = decision_quality * confidence`` per turn,
and run an outcome-consistency gate against the platform result. The older
per-turn XML-like parser is kept for compatibility with existing tests and
manual probes.

Output schema (decoded by :func:`parse_judgment`):

    <decision_quality>{value}</decision_quality>
    <judgment_tags>tag1, tag2, ...</judgment_tags>
    <rationale>...</rationale>

``decision_quality`` must be one of the seven canonical values
``{-1, -0.7, -0.3, 0, +0.3, +0.7, +1}`` (spec PDF §2). Any other value is
clamped to the nearest canonical bin during parsing — this is more
robust than strict rejection and matches the spec's "decision_quality ∈
that set" contract.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, List, Optional


CANONICAL_Q_VALUES: tuple[float, ...] = (-1.0, -0.7, -0.3, 0.0, 0.3, 0.7, 1.0)
CONFIDENCE_VALUES: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
CANONICAL_JUDGMENT_TAGS: tuple[str, ...] = (
    "goal_progress",
    "accuracy",
    "action_choice",
    "observation_use",
    "planning",
    "exploration",
    "recovery",
    "efficiency",
    "tool_or_env_protocol",
    "formatting",
    "safety_or_constraint",
    "other",
)


# ---------------------------------------------------------------------------
# Prompt construction.
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = """\
You are an expert RL trajectory judge. Given a multi-turn agent trajectory \
and one specific turn within it, you must:

1. Decide whether that one turn was a *good* or *bad* decision toward the \
agent's task goal. The goal and task description are given in the user \
message. Consider information available *only* up to that turn — do not \
penalize a turn for what happens later in the trajectory.

2. Output your judgment in the EXACT format shown below. Do not add prose \
outside the tags. Do not include code fences.

<decision_quality>VALUE</decision_quality>
<judgment_tags>tag1, tag2</judgment_tags>
<rationale>One or two sentences explaining the judgment.</rationale>

VALUE must be one of these canonical floats:
  -1.0   highly counterproductive (clearly broke the task)
  -0.7   bad (likely worsens the position)
  -0.3   slightly bad (mild inefficiency or minor mistake)
   0.0   neutral (no impact, exploration, formatting)
  +0.3   slightly helpful (small step forward)
  +0.7   good (clear progress)
  +1.0   essential (decisive correct action)

judgment_tags must be a comma-separated list drawn from:
  policy_compliance, action_choice, exploration, plan_following, \
recovery, observation_use, formatting.
Empty list is allowed.
"""


V3_SYSTEM_PROMPT = """\
You are an expert RL credit-assignment judge for long-horizon agent tasks.
Output exactly one JSON object — no markdown, no code fences, no narration.

# What you are doing
The trajectory has already finished. Your job is to identify turns where
the agent's decision materially changed the outcome (positively or
negatively), and assign a graded score to each. Use full context for
credit assignment: task goal, prior state, executed action, environment
feedback, trajectory context, and final outcome. Use the final outcome
as a consistency reference, but not as a hard rule — a successful
trajectory can still contain bad turns and a failed one can still
contain strong decisions. The downstream RL trainer uses these scores
as token-level advantage modulators — the strength and polarity of
your scores directly shape gradient updates.

# Judgment criteria
- Ground every judgment in observable evidence: environment feedback,
  score changes, error messages, task state transitions, or logical
  inference from the trajectory context, etc.
- Positive turns: advanced the goal, completed a sub-goal, set up
  necessary preconditions, recovered from an earlier mistake, gathered
  critical information, etc.
- Negative turns: hindered progress, triggered errors, repeated failed
  actions, moved away from the goal, wasted steps on irrelevant actions,
  violated environment protocol, etc.
- You may infer poor strategy even without an explicit env error — e.g.
  repeating an unsuccessful approach, ignoring relevant observations,
  choosing an obviously suboptimal path, etc.
- Skip routine/filler turns that had no meaningful impact. Only label
  turns that genuinely shifted the trajectory's outcome.

# Scoring guidance
- Commit to strong scores when the evidence is clear. ±0.7 or ±1.0
  produce meaningful gradient signal; weak scores (±0.3) are near-noise.
- Use ±0.3 only when the step's impact is genuinely ambiguous.
- For confidence: 1.0 when env feedback directly confirms the result;
  0.75 when the judgment relies on inference; 0.5 when truly uncertain.
- There is no required balance between positive and negative labels.
  Label what the evidence supports — if a trajectory is entirely bad
  decisions, label only negatives; if it has clear good steps, label
  them. Follow the evidence, not a quota.

# Output schema (every field required for emitted turns)
{
  "trajectory_outcome_recap": {"agent_succeeded": true_or_false},
  "turn_judgments": [
    {
      "turn_index": <int>,
      "decision_quality": <one of: -1.0, -0.7, -0.3, 0.3, 0.7, 1.0>,
      "confidence": <one of: 0.5, 0.75, 1.0>,
      "judgment_tags": ["<one tag>"],
      "rationale": "<concise evidence-based assessment>"
    }
  ]
}

# Rules
- Do NOT emit decision_quality = 0.0; if a turn is neutral, omit it.
- Do NOT emit confidence < 0.5; below that threshold, omit the turn.
- Keep `rationale` concise — cite specific env feedback or state changes.
- One tag per turn. Allowed tags:
  goal_progress, accuracy, action_choice, observation_use, planning,
  exploration, recovery, efficiency, tool_or_env_protocol, formatting,
  safety_or_constraint, other.
- agent_succeeded must reflect the actual final outcome (the trainer
  uses it as a consistency check; lying voids the entire labeling).

# Final reward formula (FYI)
q = decision_quality * confidence per labeled turn, applied as a
token-level advantage modulator. Strong verdicts (|q| >= 0.7) move
the policy; weak verdicts (|q| <= 0.225) are near-noise.
"""


def build_user_message(
    *,
    task_description: str,
    initial_observation: str,
    history: List[dict],          # [{"role":"assistant"|"user","content":str}]
    target_history_index: int,
    target_action: str,
    target_env_response: str,
) -> str:
    """Build the per-turn user message for the judge.

    ``history`` is the full prior dialogue *excluding* the turn under
    judgment; ``target_*`` describe the specific turn we're scoring.
    """
    lines: list[str] = []
    lines.append("# Task")
    lines.append(task_description.strip())
    lines.append("")
    if initial_observation:
        lines.append("# Initial Observation")
        lines.append(initial_observation.strip())
        lines.append("")
    if history:
        lines.append("# Prior Turns")
        for h in history:
            role = h.get("role", "")
            content = (h.get("content") or "").strip()
            if not content:
                continue
            lines.append(f"## {role}")
            lines.append(content)
        lines.append("")
    lines.append(f"# Turn To Judge (history_index = {target_history_index})")
    lines.append("## assistant")
    lines.append(target_action.strip() or "(no action)")
    lines.append("## env_response")
    lines.append(target_env_response.strip() or "(no response)")
    lines.append("")
    lines.append("Now output the judgment block in the format described in the system message.")
    return "\n".join(lines)


def build_v3_user_message(
    *,
    task_description: str,
    initial_observation: str,
    turns: List[dict],  # [{"turn_index": int, "action": str, "env_response": str}]
    task_succeeded: Optional[bool] = None,
    final_score: Optional[float] = None,
) -> str:
    """Build the trajectory-level v3 judge request."""
    lines: list[str] = []
    lines.append("# Task")
    lines.append(task_description.strip() or "(missing task description)")
    lines.append("")
    if initial_observation:
        lines.append("# Initial Observation")
        lines.append(initial_observation.strip())
        lines.append("")
    lines.append("# Full Trajectory")
    for turn in turns:
        turn_index = int(turn.get("turn_index", 0))
        action = (turn.get("action") or "").strip()
        env_response = (turn.get("env_response") or "").strip()
        lines.append(f"## Turn {turn_index}")
        lines.append("### assistant")
        lines.append(action or "(no action)")
        lines.append("### env_response")
        lines.append(env_response or "(no response)")
        if "score" in turn:
            score = int(round(float(turn["score"])))
            delta = float(turn.get("score_delta", 0.0))
            delta_str = f" (+{int(round(delta))})" if delta > 0 else ""
            lines.append(f"### score")
            lines.append(f"{score}/100{delta_str}")
    lines.append("")
    if task_succeeded is not None:
        lines.append("# Outcome")
        score_str = f", final score: {int(round(float(final_score)))}/100" if final_score is not None else ""
        lines.append(f"Task completed: {'yes' if task_succeeded else 'no'}{score_str}.")
        lines.append("")
    lines.append("Output exactly the v3 JSON object described in the system message.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parser.
# ---------------------------------------------------------------------------


_RE_DQ = re.compile(r"<decision_quality>\s*([-+]?\d*\.?\d+)\s*</decision_quality>", re.IGNORECASE)
_RE_TAGS = re.compile(r"<judgment_tags>(.*?)</judgment_tags>", re.IGNORECASE | re.DOTALL)
_RE_RAT = re.compile(r"<rationale>(.*?)</rationale>", re.IGNORECASE | re.DOTALL)


@dataclass
class Judgment:
    """Parsed judgment for one turn.

    Attributes:
        decision_quality: float in :data:`CANONICAL_Q_VALUES` (snapped to
            the nearest canonical bin).
        judgment_tags: parsed tag list (may be empty).
        rationale: free-form natural-language critique. May be empty if
            the judge omitted the tag.
        is_valid: True if all three tags were found AND decision_quality
            parsed to a finite number; False otherwise (caller should
            treat invalid judgments as ``q_t = 0`` and empty critique).
    """

    decision_quality: float
    judgment_tags: List[str]
    rationale: str
    is_valid: bool


@dataclass
class V3TurnJudgment:
    turn_index: int
    decision_quality: float
    confidence: float
    q: float
    judgment_tags: List[str]
    judgment_basis: str
    rationale: str
    is_valid: bool


@dataclass
class V3TrajectoryJudgment:
    agent_succeeded: Optional[bool]
    turns: List[V3TurnJudgment]
    is_valid: bool


def _snap_to_canonical(x: float) -> float:
    """Snap to the nearest entry in :data:`CANONICAL_Q_VALUES`."""
    return min(CANONICAL_Q_VALUES, key=lambda v: abs(v - x))


def _snap_to_confidence(x: float) -> float:
    """Snap to the nearest entry in :data:`CONFIDENCE_VALUES`."""
    return min(CONFIDENCE_VALUES, key=lambda v: abs(v - x))


_TAG_ALIASES: dict[str, str] = {
    "目标推进": "goal_progress",
    "推理准确性": "accuracy",
    "动作选择": "action_choice",
    "观测信息利用": "observation_use",
    "规划决策": "planning",
    "探索行为": "exploration",
    "纠错修复": "recovery",
    "执行效率": "efficiency",
    "工具与环境规范": "tool_or_env_protocol",
    "工具环境规范": "tool_or_env_protocol",
    "格式规范": "formatting",
    "安全约束": "safety_or_constraint",
    "其他": "other",
    "correctness": "accuracy",
    "correct": "accuracy",
    "domain_correctness": "accuracy",
    "plan_following": "planning",
    "policy_compliance": "tool_or_env_protocol",
    "env_protocol": "tool_or_env_protocol",
    "tool_protocol": "tool_or_env_protocol",
    "constraint": "safety_or_constraint",
    "safety": "safety_or_constraint",
}
_TAG_SET = set(CANONICAL_JUDGMENT_TAGS)


def _normalize_v3_judgment_tag(raw: str) -> Optional[str]:
    original = str(raw).strip()
    if not original:
        return None
    if original in _TAG_ALIASES:
        return _TAG_ALIASES[original]
    tag = original.lower().replace("-", "_").replace(" ", "_")
    tag = re.sub(r"[^a-z0-9_]+", "_", tag).strip("_")
    if not tag:
        return None
    tag = _TAG_ALIASES.get(tag, tag)
    if tag in _TAG_SET:
        return tag
    return "other"


def _normalize_v3_judgment_tags(raw_tags: list[str]) -> List[str]:
    tags: List[str] = []
    seen: set[str] = set()
    for raw_tag in raw_tags:
        tag = _normalize_v3_judgment_tag(raw_tag)
        if tag is None or tag in seen:
            continue
        tags.append(tag)
        seen.add(tag)
    return tags


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "success", "succeeded", "won"}:
            return True
        if lowered in {"false", "no", "0", "failure", "failed", "lost"}:
            return False
    return None


def _extract_json_object(raw: str) -> Optional[dict]:
    """Parse a JSON object, tolerating code fences or light surrounding prose."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def parse_judgment(raw: str) -> Judgment:
    """Parse the judge's raw text into a :class:`Judgment`.

    Robust to:
      - whitespace / newline noise inside tags
      - tags emitted in any order
      - decision_quality off the canonical grid (snaps to nearest)
      - missing optional tags (tags / rationale)

    Returns an invalid :class:`Judgment` (with ``is_valid=False``) only
    when ``decision_quality`` cannot be parsed at all — in that case the
    caller should drop this turn's annotation (q stays 0).
    """
    dq_match = _RE_DQ.search(raw)
    if dq_match is None:
        return Judgment(decision_quality=0.0, judgment_tags=[], rationale="", is_valid=False)
    try:
        dq_raw = float(dq_match.group(1))
    except ValueError:
        return Judgment(decision_quality=0.0, judgment_tags=[], rationale="", is_valid=False)
    dq = _snap_to_canonical(dq_raw)

    tags_match = _RE_TAGS.search(raw)
    tags: List[str] = []
    if tags_match:
        tags_str = tags_match.group(1).strip()
        if tags_str:
            tags = [t.strip() for t in tags_str.split(",") if t.strip()]

    rat_match = _RE_RAT.search(raw)
    rationale = rat_match.group(1).strip() if rat_match else ""

    return Judgment(
        decision_quality=dq,
        judgment_tags=tags,
        rationale=rationale,
        is_valid=True,
    )


def parse_v3_trajectory_judgment(raw: str) -> V3TrajectoryJudgment:
    """Parse tracegrpo v3 trajectory-level JSON.

    Invalid or missing per-turn fields are snapped/defaulted to neutral q=0
    rather than rejecting the entire trajectory. The caller applies the
    outcome-consistency gate because it owns the platform outcome.
    """
    data = _extract_json_object(raw)
    if data is None:
        return V3TrajectoryJudgment(agent_succeeded=None, turns=[], is_valid=False)

    recap = data.get("trajectory_outcome_recap") or {}
    agent_succeeded = _coerce_bool(recap.get("agent_succeeded"))

    raw_turns = data.get("turn_judgments") or []
    if not isinstance(raw_turns, list):
        return V3TrajectoryJudgment(
            agent_succeeded=agent_succeeded,
            turns=[],
            is_valid=False,
        )

    turns: List[V3TurnJudgment] = []
    for item in raw_turns:
        if not isinstance(item, dict):
            continue
        try:
            turn_index = int(item.get("turn_index"))
        except (TypeError, ValueError):
            continue

        is_valid = True
        try:
            dq = _snap_to_canonical(float(item.get("decision_quality")))
        except (TypeError, ValueError):
            dq = 0.0
            is_valid = False
        try:
            confidence = _snap_to_confidence(float(item.get("confidence")))
        except (TypeError, ValueError):
            confidence = 0.0
            is_valid = False

        tags_raw = item.get("judgment_tags") or []
        if isinstance(tags_raw, str):
            tags = _normalize_v3_judgment_tags([t for t in tags_raw.split(",")])
        elif isinstance(tags_raw, list):
            tags = _normalize_v3_judgment_tags([str(t) for t in tags_raw])
        else:
            tags = []

        basis = str(item.get("judgment_basis") or "").strip()
        rationale = str(item.get("rationale") or "").strip()
        q = float(dq * confidence) if is_valid else 0.0
        turns.append(
            V3TurnJudgment(
                turn_index=turn_index,
                decision_quality=float(dq),
                confidence=float(confidence),
                q=q,
                judgment_tags=tags,
                judgment_basis=basis,
                rationale=rationale,
                is_valid=is_valid,
            )
        )

    return V3TrajectoryJudgment(
        agent_succeeded=agent_succeeded,
        turns=turns,
        is_valid=True,
    )
