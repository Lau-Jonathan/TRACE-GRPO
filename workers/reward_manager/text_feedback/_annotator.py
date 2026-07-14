"""Async OpenAI-compatible client for the LLM-judge teacher.

Point the client at any OpenAI-compatible ``/v1/chat/completions`` endpoint
by setting ``TEXT_FEEDBACK_BASE_URL``, ``TEXT_FEEDBACK_MODEL``, and
``TEXT_FEEDBACK_API_KEY`` in the environment. The paper's main experiments
used a strong open-weight instruction-following LLM as the judge; any
OpenAI-compatible replacement works.

Failure model:
  - 5xx, network errors, timeouts → exponential backoff retry
    (1s, 2s, 4s with +/-50% jitter, 3 attempts)
  - 4xx (other than 429) → no retry, surface error
  - 429 → backoff retry like 5xx
  - On total failure: return ``None``; caller substitutes a neutral
    judgment (q_t = 0).
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from dataclasses import dataclass
from typing import Optional

import httpx


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Endpoint defaults — must be supplied by the caller via env vars.
# ---------------------------------------------------------------------------


def _env_nonempty(*names: str, default: str) -> str:
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        value = raw.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1].strip()
        if value:
            return value
    return default


DEFAULT_BASE_URL = _env_nonempty(
    "TEXT_FEEDBACK_BASE_URL",
    "TRACE_GRPO_LLM_BASE_URL",
    default="",
)
DEFAULT_MODEL = _env_nonempty(
    "TEXT_FEEDBACK_MODEL",
    "TRACE_GRPO_LLM_MODEL",
    default="",
)
DEFAULT_MAX_WORKERS = int(
    _env_nonempty("TEXT_FEEDBACK_MAX_WORKERS", "TRACE_GRPO_LLM_MAX_WORKERS", default="64")
)
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT_S = 120.0
REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "glm-5")


# ---------------------------------------------------------------------------
# Request / response payloads.
# ---------------------------------------------------------------------------


@dataclass
class JudgeRequest:
    """One LLM judge request (one turn)."""

    system_prompt: str
    user_message: str
    max_tokens: int = 256
    temperature: float = 0.0          # deterministic judgment
    request_id: str = ""              # opaque caller id (echoed back for ordering)


@dataclass
class JudgeResponse:
    """Result of one judge request.

    Attributes:
        request_id: same value as on the request.
        text: the assistant message body, or ``None`` if the call failed.
        error: error description (None on success).
        latency_s: wall-clock latency for this individual call.
    """

    request_id: str
    text: Optional[str]
    error: Optional[str]
    latency_s: float


# ---------------------------------------------------------------------------
# Client.
# ---------------------------------------------------------------------------


@dataclass
class LLMJudgeClient:
    """Concurrent OpenAI-compatible chat client.

    Single-instance: one ``LLMJudgeClient`` opens one ``httpx.AsyncClient``
    underneath. Use :meth:`run_batch` to fire a list of
    :class:`JudgeRequest` and gather :class:`JudgeResponse` objects.

    Args:
        base_url: full chat completions endpoint URL.
        model: model name passed in the request body.
        api_key: bearer token; ``""`` (default) skips Authorization header.
        max_workers: global concurrency cap (semaphore size).
        max_retries: number of retry attempts on transient failures.
        timeout_s: per-request HTTP timeout.
        verify_ssl: passed through to httpx; HTTP endpoints don't care.
    """

    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str = ""
    max_workers: int = DEFAULT_MAX_WORKERS
    max_retries: int = DEFAULT_MAX_RETRIES
    timeout_s: float = DEFAULT_TIMEOUT_S
    verify_ssl: bool = False

    def uses_reasoning_token_param(self) -> bool:
        """Whether this model expects max_completion_tokens instead of max_tokens."""
        model_l = (self.model or "").strip().lower()
        return any(model_l.startswith(prefix) for prefix in REASONING_MODEL_PREFIXES)

    def _headers(self) -> dict[str, str]:
        h = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _payload(self, req: JudgeRequest) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": req.system_prompt},
                {"role": "user", "content": req.user_message},
            ],
            "temperature": req.temperature,
            "stream": False,
        }
        if self.uses_reasoning_token_param():
            payload["max_completion_tokens"] = req.max_tokens
        else:
            payload["max_tokens"] = req.max_tokens
        return payload

    async def _send_one(
        self,
        client: httpx.AsyncClient,
        req: JudgeRequest,
        sem: asyncio.Semaphore,
    ) -> JudgeResponse:
        """Send one request with retry/backoff. Bound to the semaphore so
        we never exceed ``max_workers`` in flight."""
        loop = asyncio.get_event_loop()
        attempt = 0
        last_err: str = ""
        async with sem:
            while attempt < self.max_retries:
                attempt += 1
                start = loop.time()
                try:
                    resp = await client.post(
                        self.base_url,
                        headers=self._headers(),
                        json=self._payload(req),
                        timeout=self.timeout_s,
                    )
                    latency = loop.time() - start
                    if resp.status_code == 200:
                        try:
                            data = resp.json()
                            text = data["choices"][0]["message"]["content"]
                            return JudgeResponse(
                                request_id=req.request_id,
                                text=text,
                                error=None,
                                latency_s=latency,
                            )
                        except (KeyError, IndexError, ValueError) as e:
                            last_err = f"malformed response: {e}"
                            # malformed → don't retry, surface immediately
                            return JudgeResponse(
                                request_id=req.request_id,
                                text=None,
                                error=last_err,
                                latency_s=latency,
                            )
                    if resp.status_code == 429 or 500 <= resp.status_code < 600:
                        last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                        # fall through to retry
                    else:
                        # non-retryable HTTP error
                        return JudgeResponse(
                            request_id=req.request_id,
                            text=None,
                            error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                            latency_s=latency,
                        )
                except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as e:
                    last_err = f"{type(e).__name__}: {e}"
                except Exception as e:  # noqa: BLE001 — surface unknown errors with retry
                    last_err = f"{type(e).__name__}: {e}"

                # Exponential backoff with ±50% jitter (max 4 s base).
                if attempt < self.max_retries:
                    base = min(2 ** (attempt - 1), 4.0)
                    sleep_s = base * (0.5 + random.random())
                    await asyncio.sleep(sleep_s)

            return JudgeResponse(
                request_id=req.request_id,
                text=None,
                error=last_err or "max retries exceeded",
                latency_s=loop.time() - start,
            )

    async def run_batch(self, requests: list[JudgeRequest]) -> list[JudgeResponse]:
        """Fire all requests concurrently (capped at ``max_workers``) and
        return responses in input order."""
        if not requests:
            return []
        sem = asyncio.Semaphore(self.max_workers)
        async with httpx.AsyncClient(verify=self.verify_ssl) as client:
            tasks = [self._send_one(client, r, sem) for r in requests]
            return await asyncio.gather(*tasks)

    def run_batch_sync(self, requests: list[JudgeRequest]) -> list[JudgeResponse]:
        """Synchronous façade — convenient for verl's reward-manager hook
        which runs on the trainer driver process (no event loop)."""
        return asyncio.run(self.run_batch(requests))
