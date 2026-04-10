"""OpenAI-compatible chat completion with parse/retry."""

from __future__ import annotations

import os
import re
import time
from typing import Any, List, Optional, Tuple

from simulation.questions_io import Question, validate_option_id


def _strip_first_line(text: str) -> Tuple[str, str]:
    text = (text or "").strip()
    if not text:
        return "", ""
    lines = text.splitlines()
    first = lines[0].strip()
    rest = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
    # Allow model to wrap id in quotes or add trailing period
    first = first.strip("\"'")
    first = re.sub(r"\s+", "", first)
    if first.endswith("."):
        first = first[:-1]
    return first, rest


def parse_response(raw: str, question: Question) -> Tuple[Optional[str], str]:
    option_id, reasoning = _strip_first_line(raw)
    if not option_id:
        return None, reasoning
    if validate_option_id(question, option_id):
        return option_id, reasoning
    return None, reasoning


def get_api_key(prefer: str = "openai") -> Optional[str]:
    if prefer == "together":
        return os.environ.get("TOGETHER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    return os.environ.get("OPENAI_API_KEY")


def make_client(
    *,
    api_key: Optional[str],
    base_url: Optional[str],
):
    """Return OpenAI client or None if no key."""
    if not api_key:
        return None
    from openai import OpenAI

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def call_llm(
    client: Any,
    *,
    system: str,
    user: str,
    model: str,
    temperature: float,
    question: Question,
    max_attempts: int = 3,
    temperatures: Optional[List[float]] = None,
) -> Tuple[str, Optional[str], str, int, Optional[str]]:
    """
    Returns (raw_full_text, parsed_option_id, reasoning, latency_ms, error).
    Retries on parse failure with increasing temperatures.
    """
    if client is None:
        return "", None, "", 0, "no_client"

    if temperatures is None:
        temperatures = [temperature, min(temperature + 0.3, 0.9), min(temperature + 0.6, 1.0)]

    last_raw = ""
    last_err: Optional[str] = None

    for attempt, temp in enumerate(temperatures[:max_attempts]):
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=float(temp),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            raw = (resp.choices[0].message.content or "").strip()
            last_raw = raw
            lat = int((time.perf_counter() - t0) * 1000)
            opt, reason = parse_response(raw, question)
            if opt:
                return raw, opt, reason, lat, None
            last_err = "parse_failed"
        except Exception as exc:  # noqa: BLE001
            lat = int((time.perf_counter() - t0) * 1000)
            last_raw = ""
            last_err = str(exc)
            if attempt == max_attempts - 1:
                return last_raw, None, "", lat, last_err

    return last_raw, None, "", lat, last_err or "parse_failed"
