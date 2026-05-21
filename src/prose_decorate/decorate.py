"""Single-chunk Claude decoration with safety guards.

Per-chunk path: build prompt -> Anthropic call -> validate -> return.
Failure paths: 3 retries, then passthrough that chunk so the pipeline
keeps producing audio.

Safety guards (per v3 spec):
- exact-equality: `strip_tags(output) == normalize_whitespace(input)`
- tag-overhead cap: `len(output) - len(strip_tags(output)) <= 0.5 * len(input)`
- nonce-collision: if prev_context already contains the nonce literal,
  treat as a hard error -> passthrough.
"""
from __future__ import annotations

import difflib
import hashlib
import os
import re
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .markdown_strip import _fold_typography, strip_tags


# Word = run of alphanumeric chars, case-folded for tolerance against
# stylistic capitalization (e.g. "the" vs "The" at sentence boundaries
# the LLM may rewrap). Apostrophes inside contractions are part of the
# token — `don't` is one word, not "don" + "t".
_WORD_RE = re.compile(r"[\w']+", re.UNICODE)


def _word_tokens(text: str) -> list[str]:
    return _WORD_RE.findall(_fold_typography(text).casefold())


def _word_drift_summary(expected: list[str], actual: list[str]) -> str:
    """Compact summary of how the word streams differ — enough to
    diagnose from a stderr line without spamming the journal."""
    delta = list(difflib.ndiff(expected, actual))
    removed = [d[2:] for d in delta if d.startswith("- ")]
    added = [d[2:] for d in delta if d.startswith("+ ")]
    n_exp, n_act = len(expected), len(actual)
    parts = [f"expected={n_exp} actual={n_act}"]
    if removed:
        parts.append(f"missing={removed[:5]!r}{'...' if len(removed) > 5 else ''}")
    if added:
        parts.append(f"added={added[:5]!r}{'...' if len(added) > 5 else ''}")
    return " ".join(parts)

# Cache-invalidating constants. Bump anything here when changing the
# Anthropic SDK floor or the API-version header the client sends.
DEFAULT_MODEL = "claude-sonnet-4-6"
ANTHROPIC_API_VERSION = "2023-06-01"  # the messages API header value
DEFAULT_MAX_TOKENS = 8192


_PROMPT_TEMPLATE = """\
You are inserting prosody markers into prose for Fish Audio's S2-Pro
text-to-speech model. S2-Pro accepts inline `[bracket]` tags written
as free-form natural language; the model uses the bracketed phrase to
shape the delivery of the words that follow.

## Tag reference

Pauses:
  [short pause]   — comma-length beat (around 250ms)
  [long pause]    — paragraph-break beat (around 800ms)
Emphasis:
  [emphasis]      — stress the next word or short phrase
Tone (use sparingly; only when the prose ITSELF establishes the tone):
  [thoughtfully] [softly] [firmly] [warmly] [drily] [reading aloud]
  [reflectively] [matter-of-factly]
Non-verbal (only if the prose explicitly invites it):
  [inhale] [exhale] [clears throat] [sigh] [chuckle]

Tags are NOT a fixed vocabulary. Any short natural-language description
in brackets works (e.g. `[after a moment of hesitation]`). Prefer the
short standard tags listed above unless the prose is doing something
unusual.

## Rules

1. PRESERVE EVERY WORD VERBATIM. INSERT tags, NEVER rewrite or delete
   prose. Don't paraphrase, don't fix typos, don't reflow lines.
2. Use tags SPARINGLY. Most prose needs zero or one tag per paragraph.
   The narrator's default cadence is already good — tags exist for the
   moments where it ISN'T.
3. DO NOT echo the `<ctx-prev-*>` block. It's read-only context for
   tonal continuity, not part of the chunk you decorate. Output only
   the decorated CURRENT chunk.
4. OUTPUT FORMAT: plain text with `[bracket]` tags inline. No markdown
   syntax (no `**bold**`, no `_italic_`, no `#`, no `>`). If you see
   bold/italic in the input, decide whether to mark it with a tag and
   then drop the syntax characters. Don't pass markdown through.
5. Map cues from the input judiciously, not mechanically:
   - A heading often deserves a `[long pause]` afterwards but not always.
   - A `**bolded**` word might warrant `[emphasis]` but most bold text
     in articles is technical-term marking, not emotional emphasis.
     Skip it unless the surrounding prose clearly calls for stress.
   - An em-dash can be a long pause OR a short aside-pause-then-resume.
     Read the sentence and decide. Don't tag every em-dash.
   - A blockquote is usually `[reading aloud]` or `[thoughtfully]`. Tag
     the FIRST sentence of the quote, not every line.

## Example

INPUT:
# Cats Sleep A Lot

Cats sleep up to **sixteen** hours a day. The exact number depends on
age, diet, and the quality of available sunbeams.

> Older cats sleep more — kittens, surprisingly, also sleep a lot.

OUTPUT:
Cats Sleep A Lot [long pause] Cats sleep up to [emphasis] sixteen \
hours a day. The exact number depends on age, diet, and the quality \
of available sunbeams. [thoughtfully] Older cats sleep more [short \
pause] kittens, surprisingly, also sleep a lot.

Decorate the chunk below.
"""


def prompt_template_hash() -> str:
    return hashlib.sha256(_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()


# Decoration result types -------------------------------------------------

class DecorateStatus:
    DECORATED = "decorated"
    PASSTHROUGH = "passthrough"


@dataclass(frozen=True)
class DecorateResult:
    text: str
    status: str
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status == DecorateStatus.DECORATED


# Key resolution ----------------------------------------------------------

class MissingAPIKey(RuntimeError):
    pass


def read_api_key() -> str:
    file_var = os.environ.get("ANTHROPIC_API_KEY_FILE", "").strip()
    if file_var:
        return Path(file_var).read_text(encoding="utf-8").strip()
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise MissingAPIKey(
            "set ANTHROPIC_API_KEY or ANTHROPIC_API_KEY_FILE "
            "(see .env.example; on dellan, agenix secret anthropic-api-key)"
        )
    return key


def make_client(api_key: str, *, timeout: float = 120.0) -> Any:
    import anthropic
    return anthropic.Anthropic(api_key=api_key, timeout=timeout)


# Single-chunk decoration -------------------------------------------------

def _build_user_message(chunk_text: str, prev_context: str) -> tuple[str, str]:
    """Returns (user_message, nonce). Nonce-delimited prev-context defeats
    any literal `<ctx-prev-...>` content in the chunk from being treated
    as a tag boundary by the model.
    """
    nonce = secrets.token_hex(8)  # 16 hex chars
    open_tag = f"<ctx-prev-{nonce}>"
    close_tag = f"</ctx-prev-{nonce}>"

    if prev_context and (open_tag in chunk_text or close_tag in chunk_text):
        # Should be astronomically rare but the spec calls this a hard
        # error rather than silent handling.
        raise NonceCollision(
            f"nonce {nonce!r} present in chunk; bailing for safety"
        )

    parts: list[str] = []
    if prev_context:
        # Escape `<` in payload so the model can't be tricked into
        # closing the delimiter early.
        escaped = prev_context.replace("<", "&lt;")
        parts.append(f"{open_tag}\n{escaped}\n{close_tag}\n")
    parts.append("CHUNK TO DECORATE:\n")
    parts.append(chunk_text)
    return "\n".join(parts), nonce


class NonceCollision(RuntimeError):
    pass


def _validate(input_text: str, output_text: str) -> tuple[bool, str]:
    """Apply safety guards from the v3 spec. Returns (ok, reason).

    The drift check compares at the WORD level, not char level. The
    system prompt's contract is "preserve every word verbatim, insert
    tags only" — char-level equality false-positives on every minor
    reasonable rewrite the LLM does (smart-quote folding, markdown-
    emphasis drop, dedup-of-doubled-title-line, space inserted between
    run-together-sentences). Sonnet at temperature=0 still does these.
    Comparing as a sequence of word tokens (alphanumeric runs) catches
    actual prose drift (added/dropped/changed words) and tolerates the
    rest.
    """
    expected_words = _word_tokens(input_text)
    actual_words = _word_tokens(strip_tags(output_text))
    if expected_words != actual_words:
        diff = _word_drift_summary(expected_words, actual_words)
        return False, f"prose drift at word level: {diff}"

    # Tag-overhead budget: 50% of input chars or 200 chars, whichever is
    # larger. The floor protects tiny inputs ("Hello world." is 12 chars
    # but a single `[short pause]` is 13 — without the floor, every tag
    # would trip the cap on a short chunk).
    tag_chars = len(output_text) - len(strip_tags(output_text))
    budget = max(0.5 * len(input_text), 200)
    if tag_chars > budget:
        return False, f"tag overhead too high ({tag_chars} > {int(budget)})"

    return True, ""


def _retriable(exc: BaseException) -> bool:
    """Errors worth retrying with backoff. Imported lazily to keep
    anthropic optional at import time (tests can mock the client without
    importing the SDK)."""
    try:
        import anthropic
    except ImportError:
        return False
    return isinstance(
        exc,
        (
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.InternalServerError,
            anthropic.RateLimitError,
        ),
    )


def decorate_chunk(
    client: Any,
    chunk_text: str,
    *,
    prev_context: str = "",
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_retries: int = 3,
    sleep: Callable[[float], None] = time.sleep,
) -> DecorateResult:
    """Decorate one chunk. On any failure path returns a passthrough
    result containing the unmodified chunk_text so callers can keep the
    pipeline moving (and surface the exit-code-10 signal at the CLI)."""
    try:
        user_msg, _ = _build_user_message(chunk_text, prev_context)
    except NonceCollision as e:
        return DecorateResult(
            text=chunk_text, status=DecorateStatus.PASSTHROUGH, reason=str(e)
        )

    backoffs = [2.0, 4.0, 8.0][:max_retries]
    last_err: Exception | None = None

    for delay in [0.0, *backoffs]:
        if delay > 0:
            sleep(delay)
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0,
                system=_PROMPT_TEMPLATE,
                messages=[{"role": "user", "content": user_msg}],
            )
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if _retriable(exc):
                continue
            return DecorateResult(
                text=chunk_text,
                status=DecorateStatus.PASSTHROUGH,
                reason=f"non-retriable: {exc}",
            )

        # Anthropic SDK content is a list of blocks; first text block wins.
        # If the model returned no text, treat as a retriable failure.
        try:
            output = "".join(
                block.text for block in resp.content if getattr(block, "type", "") == "text"
            )
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue

        if not output.strip():
            last_err = RuntimeError("empty response")
            continue

        ok, reason = _validate(chunk_text, output)
        if ok:
            return DecorateResult(text=output, status=DecorateStatus.DECORATED)
        last_err = RuntimeError(reason)
        # Validation failure isn't transient — re-prompting at temperature=0
        # would just give us the same bad output. Fall through to passthrough.
        break

    return DecorateResult(
        text=chunk_text,
        status=DecorateStatus.PASSTHROUGH,
        reason=str(last_err) if last_err else "unknown",
    )
