"""Gemini multimodal audio decoration path.

Mirrors decorate.py::decorate_chunk shape — same DecorateResult union,
same retry / passthrough discipline, same _validate guards — but reads
GEMINI_API_KEY (not ANTHROPIC_API_KEY) and routes the chunk through the
google-genai SDK with the actual audio bytes inlined alongside the
transcript chunk. The model is told to base every prosody decision on
what it HEARS in the audio, not on inference from the prose alone.

Why a separate module instead of a branch inside decorate.py: the
provider construction (api_key shape, request body, response shape) is
different enough that interleaving the two paths would smear the
prompt-template / validation logic across both. Keeping the audio path
parallel makes the diff easy to read and the rollback (delete this
file + the CLI flag) one-step.

Cache key partitioning: the CLI computes `sha256(audio_chunk_bytes)`
and passes it as `audio_hash` to `cache.key_for`. That key is disjoint
from the text-only key for the same chunk — flipping between
text-only and audio-grounded never serves a stale entry from the
wrong path.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any, Callable

from .decorate import (
    DecorateResult,
    DecorateStatus,
    _build_system_prompt as _build_text_system_prompt,
    _validate,
)


# Gemini 2.5 Pro picked over 2.5 Flash for the long-form-audio path:
# the audio understanding gap matters more here than the latency
# saving — we're doing per-chunk prosody calls offline anyway, not
# interactive. Override via PROSE_DECORATE_AUDIO_MODEL when iterating.
AUDIO_DEFAULT_MODEL = "gemini-2.5-pro"
DEFAULT_MAX_OUTPUT_TOKENS = 8192

# Inline-data is the SDK's lightweight path: bytes go in the request
# body, no File API upload step. Gemini's documented inline cap is
# 20 MB per request; we apply a soft cap below that to leave headroom
# for the system prompt + chunk text + response budget. Anything
# bigger should chunk first via prose_decorate.audio_chunk (silence-
# aligned with the transcript).
INLINE_AUDIO_SOFT_CAP_BYTES = 18 * 1024 * 1024

# Audio MIME types Gemini accepts. We default to audio/mp3 because the
# input pipeline (Auphonic clone-enhance step in tts-tool / Substack
# article audio) lands as MP3; callers can override per-chunk.
DEFAULT_AUDIO_MIME = "audio/mp3"


# Suffix -> MIME mapping for inline audio requests. `.opus` lives here
# because podcast-side pipelines (and Telegram voice notes) commonly
# emit it, and Gemini rejects a `.opus` payload sent under audio/mp3.
# Callers fall back to DEFAULT_AUDIO_MIME for unknown suffixes — that
# gives a useful error on the Gemini side rather than a silent crash.
AUDIO_MIME_BY_EXT: dict[str, str] = {
    ".mp3": "audio/mp3",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".aac": "audio/aac",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".flac": "audio/flac",
}


_AUDIO_GROUNDING_ADDENDUM = """\

## Audio-grounded section (IMPORTANT — applies to THIS request only)

You have been given a recording of the narrator delivering the prose
chunk below. Base EVERY prosody tag decision on what you HEAR in that
audio — the actual pauses, breath, tone shifts, and emphasis the
narrator produced — NOT on inferences from the prose alone.

Specifically:
- A pause goes where the narrator audibly paused, sized to the
  pause's real duration (`[short pause]` for ~250ms, `[long pause]`
  for ~800ms, `[N second pause]` for measured durations beyond that).
- A `[emphasis]` goes where the narrator audibly stressed a word —
  louder, higher pitch, longer vowel — NOT where the prose merely
  "looks like" it would be emphasized.
- A tonal voice opener (`[softly]`, `[firmly]`, etc. from the closed
  set) goes where the narrator's voice audibly shifted register.
  Close it with `[back to narration]` when the register returns.
- A non-verbal cue (`[inhale]`, `[chuckle]`, etc.) goes only when the
  narrator audibly produced one.

If the narrator's delivery is flat / unmarked through a span, output
that span with NO tags. Empty tags are worse than no tags. The model
that consumes your output (Fish s2-pro) will fall back to its
default cadence and produce listenable audio either way.

The text chunk below is what the narrator says (verbatim — preserve
every word). Output the same words, with bracket tags inserted at
the audible-prosody-event positions.
"""


class MissingGeminiAPIKey(RuntimeError):
    pass


class MissingGoogleGenAI(RuntimeError):
    pass


def read_gemini_api_key() -> str:
    """Resolve the Gemini API key. Prefers GEMINI_API_KEY_FILE (set by
    the nixos-config writeShellApplication wrapper that points at the
    agenix decrypt path) over GEMINI_API_KEY (bare env var for local
    dev / CI). Raises MissingGeminiAPIKey with a clear message when
    neither is set."""
    file_var = os.environ.get("GEMINI_API_KEY_FILE", "").strip()
    if file_var:
        return Path(file_var).read_text(encoding="utf-8").strip()
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise MissingGeminiAPIKey(
            "set GEMINI_API_KEY or GEMINI_API_KEY_FILE "
            "(on dellan, agenix secret gemini-api-key)"
        )
    return key


def audio_hash_for(audio_bytes: bytes) -> str:
    """sha256 hex of the audio chunk bytes. Used as the
    `audio_hash` partition key in `cache.key_for` so an audio run
    never serves a text-only cache hit for the same chunk text."""
    return hashlib.sha256(audio_bytes).hexdigest()


def make_client(api_key: str) -> Any:
    """Construct a google-genai client. Imported lazily so the
    anthropic-only path doesn't require google-genai at import
    time — unit tests mock the client object and never touch this
    function."""
    try:
        from google import genai  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MissingGoogleGenAI(
            "google-genai is required for --audio; install via "
            "`pip install google-genai` or add it to your flake's "
            "Python dep set"
        ) from exc
    return genai.Client(api_key=api_key)


def _build_audio_system_prompt(register: str = "") -> str:
    """Reuse the text-path system prompt (so tag vocabulary,
    word-preservation rules, register preamble all stay in sync) and
    append the audio-grounding addendum that tells the model to base
    decisions on what it heard, not what it inferred from prose."""
    return _build_text_system_prompt(register) + _AUDIO_GROUNDING_ADDENDUM


def _build_contents(chunk_text: str, audio_bytes: bytes, audio_mime: str) -> list[Any]:
    """Build the `contents` list for client.models.generate_content.

    Two parts: the audio (so the model anchors on it FIRST — Gemini's
    documented ordering preference is media-then-text for grounded
    extraction tasks) and the chunk text. Imported lazily.

    Falls back to a plain dict structure if google-genai types aren't
    importable, which only happens in unit tests; the mock client
    accepts whatever we hand it and asserts on the bytes / text
    payload, not on the SDK class identity.
    """
    try:
        from google.genai import types  # type: ignore[import-not-found]
        audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=audio_mime)
        text_part = types.Part.from_text(text=f"CHUNK TO DECORATE:\n{chunk_text}")
        return [audio_part, text_part]
    except ImportError:
        # Test-friendly fallback. Mirrors the inline-data shape so
        # downstream assertions still find the bytes.
        return [
            {"inline_data": {"mime_type": audio_mime, "data": audio_bytes}},
            {"text": f"CHUNK TO DECORATE:\n{chunk_text}"},
        ]


def _build_config(register: str, max_output_tokens: int) -> Any:
    """Build types.GenerateContentConfig with system_instruction +
    deterministic temperature. Falls back to a plain SimpleNamespace
    for unit tests."""
    sysi = _build_audio_system_prompt(register)
    try:
        from google.genai import types  # type: ignore[import-not-found]
        return types.GenerateContentConfig(
            system_instruction=sysi,
            temperature=0,
            max_output_tokens=max_output_tokens,
        )
    except ImportError:
        from types import SimpleNamespace
        return SimpleNamespace(
            system_instruction=sysi,
            temperature=0,
            max_output_tokens=max_output_tokens,
        )


def _retriable_audio(exc: BaseException) -> bool:
    """Errors worth retrying with backoff.

    Two layers of recognition:

    1. Transport-level exceptions raised BEFORE the google-genai SDK
       wraps them — `httpx.HTTPError` covers timeouts / connect resets,
       and the stdlib `TimeoutError` / `ConnectionError` catch any
       layer that surfaces those directly. These are always retriable
       and work even when the google-genai SDK isn't importable (test
       envs).
    2. SDK-wrapped `google.genai.errors.APIError` with a 429 or 5xx
       `code` / `status_code`. Auth / invalid-argument (4xx other
       than 429) are terminal.

    Anything not matched is treated as non-retriable (same shape as
    `decorate.py::_retriable`).
    """
    # Layer 1: transport / stdlib errors that don't depend on the SDK.
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    try:
        import httpx  # type: ignore[import-not-found]
        if isinstance(exc, httpx.HTTPError):
            return True
    except ImportError:
        pass

    # Layer 2: SDK-wrapped API errors. Importing google.genai.errors
    # may itself fail in stripped-down test envs; in that case we've
    # already returned False from layer 1's miss.
    try:
        from google.genai import errors as genai_errors  # type: ignore[import-not-found]
    except ImportError:
        return False
    if isinstance(exc, genai_errors.APIError):
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if code is None:
            return True  # unknown error code -> assume transient
        try:
            code_int = int(code)
        except (TypeError, ValueError):
            return True
        return code_int == 429 or 500 <= code_int < 600
    return False


def decorate_chunk_with_audio(
    client: Any,
    chunk_text: str,
    *,
    audio_bytes: bytes,
    audio_mime: str = DEFAULT_AUDIO_MIME,
    model: str = AUDIO_DEFAULT_MODEL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    max_retries: int = 3,
    strict_tones: bool = False,
    register: str = "",
    sleep: Callable[[float], None] = time.sleep,
) -> DecorateResult:
    """Decorate one chunk against its matching audio segment.

    On ANY failure path (transport, validation, empty response) returns
    a passthrough result containing the unmodified chunk_text so the
    pipeline keeps producing audio — same discipline as the Claude
    text-only path.
    """
    contents = _build_contents(chunk_text, audio_bytes, audio_mime)
    config = _build_config(register, max_output_tokens)

    backoffs = [2.0, 4.0, 8.0][:max_retries]
    last_err: Exception | None = None

    for delay in [0.0, *backoffs]:
        if delay > 0:
            sleep(delay)
        try:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if _retriable_audio(exc):
                continue
            return DecorateResult(
                text=chunk_text,
                status=DecorateStatus.PASSTHROUGH,
                reason=f"non-retriable: {exc}",
            )

        # response.text on google-genai is a str (or None when all
        # candidates were blocked / empty). Empty / None -> retriable.
        output = getattr(resp, "text", None)
        if not output or not output.strip():
            last_err = RuntimeError("empty response")
            continue

        ok, reason = _validate(chunk_text, output, strict_tones=strict_tones)
        if ok:
            return DecorateResult(text=output, status=DecorateStatus.DECORATED)
        last_err = RuntimeError(reason)
        # Validation failure at temperature=0 is sticky; re-prompting
        # would just regenerate the same bad output. Fall through.
        break

    return DecorateResult(
        text=chunk_text,
        status=DecorateStatus.PASSTHROUGH,
        reason=str(last_err) if last_err else "unknown",
    )
