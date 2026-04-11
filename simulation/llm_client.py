"""LLM chat completion (Together native SDK or OpenAI-compatible client) with parse/retry."""

from __future__ import annotations

import os
import re
import time
import json
from typing import Any, Dict, List, Optional, Tuple

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


def _extract_json_blob(raw: str) -> str:
    t = (raw or "").strip()
    if "```" in t:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", t, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return t


def _strip_markdown_fence_prefix(raw: str) -> str:
    """If the model opened a fenced block but never closed it, drop the fence line only."""
    t = (raw or "").strip()
    if not t:
        return ""
    t = re.sub(r"^```(?:json)?\s*", "", t, count=1, flags=re.IGNORECASE)
    return t.strip()


def _find_matching_brace(s: str, open_idx: int) -> int:
    """
    ``open_idx`` must point at ``{``. Return index *after* the matching ``}``, or ``-1`` if incomplete.
    Respects string literals so braces inside reasoning text are ignored.
    """
    if open_idx >= len(s) or s[open_idx] != "{":
        return -1
    depth = 0
    i = open_idx
    in_str = False
    while i < len(s):
        c = s[i]
        if in_str:
            if c == "\\" and i + 1 < len(s):
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            i += 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def _answers_body_for_salvage(text: str) -> str:
    """
    Prefer the object value of ``answers`` so we do not match ``question_id`` strings inside other keys.
    If ``answers`` is truncated mid-object, return from its opening ``{`` to end of ``text``.
    """
    m = re.search(r'"answers"\s*:\s*\{', text)
    if not m:
        return text
    start = m.end() - 1
    end = _find_matching_brace(text, start)
    if end < 0:
        return text[start:]
    return text[start:end]


def _collect_valid_cells_from_answers_map(
    answers: Dict[str, Any],
    questions: List[Question],
) -> Dict[str, Dict[str, Any]]:
    """Return only question cells that are structurally valid (partial map allowed)."""
    out: Dict[str, Dict[str, Any]] = {}
    for q in questions:
        cell = answers.get(q.question_id)
        if not isinstance(cell, dict):
            continue
        oid = cell.get("option_id")
        if oid is not None:
            oid = str(oid).strip().strip("\"'")
        reason = str(cell.get("reasoning", "") or "").strip()
        if not oid or not validate_option_id(q, oid):
            continue
        out[q.question_id] = {"option_id": oid, "reasoning": reason}
    return out


def _salvage_partial_answers_from_text(text: str, questions: List[Question]) -> Dict[str, Dict[str, Any]]:
    """
    When ``json.loads`` fails (truncated stream, unclosed fence), pull per-question objects by key + brace scan.
    """
    body = _answers_body_for_salvage(text)
    out: Dict[str, Dict[str, Any]] = {}
    for q in questions:
        qid = q.question_id
        needle = f'"{qid}"'
        idx = 0
        while True:
            pos = body.find(needle, idx)
            if pos < 0:
                break
            j = pos + len(needle)
            while j < len(body) and body[j].isspace():
                j += 1
            if j < len(body) and body[j] == ":":
                j += 1
            while j < len(body) and body[j].isspace():
                j += 1
            if j >= len(body) or body[j] != "{":
                idx = pos + 1
                continue
            end = _find_matching_brace(body, j)
            if end < 0:
                idx = pos + 1
                continue
            frag = body[j:end]
            try:
                cell = json.loads(frag)
            except json.JSONDecodeError:
                idx = pos + 1
                continue
            if not isinstance(cell, dict):
                idx = pos + 1
                continue
            oid = cell.get("option_id")
            if oid is not None:
                oid = str(oid).strip().strip("\"'")
            reason = str(cell.get("reasoning", "") or "").strip()
            if oid and validate_option_id(q, oid):
                out[qid] = {"option_id": oid, "reasoning": reason}
                break
            idx = pos + 1
    return out


def parse_survey_json(raw: str, questions: List[Question]) -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    """
    Parse ``{"answers": {qid: {"option_id": "...", "reasoning": "..."}}}``.

    Returns ``(per_question_cell, error_message)``. On truncated or invalid JSON, still returns any
    question blocks that could be validated (partial fill). ``error_message`` is ``None`` only when
    every question in ``questions`` is present and valid.
    """
    nq = len(questions)
    blob = _extract_json_blob(raw)
    decode_err: Optional[str] = None
    data: Optional[dict] = None
    if blob.strip():
        try:
            loaded = json.loads(blob)
            data = loaded if isinstance(loaded, dict) else None
        except json.JSONDecodeError as exc:
            decode_err = f"json_decode:{exc}"
    else:
        decode_err = "empty_json_blob"

    if isinstance(data, dict):
        answers = data.get("answers")
        if isinstance(answers, dict):
            out = _collect_valid_cells_from_answers_map(answers, questions)
            if len(out) == nq:
                return out, None
            if len(out) > 0:
                return out, f"partial_survey_parse:{len(out)}/{nq}"

    stripped = _strip_markdown_fence_prefix(raw)
    salvage_sources = [blob, stripped, raw.strip()]
    seen: set[str] = set()
    best: Dict[str, Dict[str, Any]] = {}
    for src in salvage_sources:
        if not src or src in seen:
            continue
        seen.add(src)
        body = _answers_body_for_salvage(src)
        cand = _salvage_partial_answers_from_text(body, questions)
        if len(cand) > len(best):
            best = cand
    if len(best) == nq:
        return best, None
    if len(best) > 0:
        return best, f"partial_survey_parse:{len(best)}/{nq}"

    return {}, decode_err or "parse_survey_json_failed"


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
    provider: str = "together",
):
    """
    Return a chat client or None if no key.

    - provider ``together``: Together serverless SDK (``from together import Together``).
    - provider ``openai``: ``openai.OpenAI``, optional ``base_url`` for compatible endpoints.
    """
    if not api_key:
        return None
    if provider == "together":
        from together import Together

        return Together(api_key=api_key)
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


def call_llm_survey_json(
    client: Any,
    *,
    system: str,
    user: str,
    model: str,
    temperature: float,
    questions: List[Question],
    max_attempts: int = 3,
    temperatures: Optional[List[float]] = None,
) -> Tuple[str, Dict[str, Dict[str, Any]], int, Optional[str]]:
    """
    One completion for the full survey. Returns (raw_text, answers_by_question_id, latency_ms, error).
    """
    if client is None:
        return "", {}, 0, "no_client"

    if temperatures is None:
        temperatures = [temperature, min(temperature + 0.3, 0.9), min(temperature + 0.6, 1.0)]

    last_raw = ""
    last_err: Optional[str] = None
    best_parsed: Dict[str, Dict[str, Any]] = {}
    best_raw = ""
    best_lat = 0

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
            parsed, err = parse_survey_json(raw, questions)
            nq = len(questions)
            if len(parsed) == nq:
                return raw, parsed, lat, None
            if len(parsed) > len(best_parsed):
                best_parsed = parsed
                best_raw = raw
                best_lat = lat
            if not parsed:
                last_err = err or "parse_failed"
        except Exception as exc:  # noqa: BLE001
            lat = int((time.perf_counter() - t0) * 1000)
            last_raw = ""
            last_err = str(exc)
            if attempt == max_attempts - 1:
                return last_raw, {}, lat, last_err

    if best_parsed:
        return best_raw, best_parsed, best_lat, None
    return last_raw, {}, lat, last_err or "parse_failed"
