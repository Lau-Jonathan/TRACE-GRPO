"""Parser for the BEACON-style ``<think>...</think><action>...</action>``
assistant response format.

TraceGRPO parser core (the paper §4.2) uses the same tag structure:

    _ACTION_RE = re.compile(r"<action>\\s*(.*?)\\s*</action>", re.DOTALL | re.IGNORECASE)
    _THINK_RE  = re.compile(r"<think>\\s*(.*?)\\s*</think>",   re.DOTALL | re.IGNORECASE)

    def _parse_action(response_text):
        m = _ACTION_RE.search(response_text)
        if not m:
            return None, True             # missing <action>
        if not _THINK_RE.search(response_text):
            return m.group(1).strip(), True   # missing <think>
        return m.group(1).strip(), False

A turn with ``has_format_error=True`` feeds ``response_text[-20:]`` to the
env (BEACON ``projection.py`` convention) so the env returns
"no known action matches" and the trajectory continues without spurious
progress. BEACON's ScienceWorld projection is case-sensitive on these tags
and additionally marks any raw response containing Chinese characters as invalid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL)
_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)


@dataclass
class ParsedAction:
    """Result of :func:`parse_action`.

    Attributes:
        action: the extracted action string (only meaningful when
            ``has_format_error`` is False; an explicit ``None`` indicates
            the response had no ``<action>`` tag at all).
        has_format_error: True if either ``<action>`` or ``<think>`` tag
            was missing.
        env_input: the string to feed env.step(); equals ``action`` when
            valid, else the BEACON fallback ``response_text[-20:]``.
    """

    action: str | None
    has_format_error: bool
    env_input: str


def parse_action(response_text: str) -> ParsedAction:
    """Extract action from a ``<think>...</think><action>...</action>``
    response.

    BEACON convention: missing-action turns feed the last 20 chars to env
    so it produces "no known action matches" and the trajectory does not
    silently advance the score. BEACON also treats Chinese characters in
    the raw response as an invalid action-format signal.
    """
    if response_text is None:
        return ParsedAction(action=None, has_format_error=True, env_input="")
    m = _ACTION_RE.search(response_text)
    has_chinese = re.search(r"[\u4e00-\u9fff]", response_text) is not None
    if m is None:
        return ParsedAction(
            action=None,
            has_format_error=True,
            env_input=response_text[-20:] if response_text else "",
        )
    action = m.group(1).strip()
    if not action:
        # Empty <action></action> — model can otherwise spam these to dodge
        # the format-error penalty without consuming env steps. Treat as
        # format error and feed BEACON's last-20-char fallback to env so
        # env.step() still runs and "no known action" advances the step
        # counter.
        return ParsedAction(
            action="",
            has_format_error=True,
            env_input=response_text[-20:] if response_text else "",
        )
    if _THINK_RE.search(response_text) is None:
        # Action present but no <think> — still flagged as format error per spec.
        return ParsedAction(action=action, has_format_error=True, env_input=action)
    if has_chinese:
        return ParsedAction(action=action, has_format_error=True, env_input=action)
    return ParsedAction(action=action, has_format_error=False, env_input=action)
