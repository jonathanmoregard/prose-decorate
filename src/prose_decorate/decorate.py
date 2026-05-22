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


# Tonal voice tags PERSIST forward in S2-Pro. The prompt restricts the
# LLM to a closed set of openers so the validator can enforce close-
# tag pairing without false-positiving on transient adverbial cues
# (e.g. `[after a moment]`, `[briefly]`) that AREN'T persistent voice
# switches. Keep this set in sync with the "Tonal voice tags" section
# of _PROMPT_TEMPLATE.
_TONE_OPENERS = frozenset({
    "reading aloud",
    "thoughtfully",
    "softly",
    "firmly",
    "warmly",
    "drily",
    "reflectively",
    "matter-of-factly",
})

# Closer detection is substring-based so minor phrasing variants Sonnet
# emits still count as a close. Phrases are anchored on the *return-to-
# narration* intent so a descriptive tag like `[narrator pauses]` or
# `[the narrator's tone shifts]` is NOT misclassified as a closer
# (round-2 advisor M1). Each entry below is a multi-word phrase that
# only makes sense as "voice returns to default".
#
# Entries MUST be lowercase — they're matched against a casefolded
# body, so any mixed-case literal here would silently never match.
_TONE_CLOSER_SUBSTRINGS = (
    "back to narration",
    "back to the narrator",
    "back to the narrator's voice",
    "narrator's voice",
    "narrator returns",
    "resuming narration",
    "default voice",
    "regular voice",
    "normal voice",
)

_TAG_EXTRACT_RE = re.compile(r"\[([^\[\]\n]+)\]")


def _classify_tag(tag_body: str) -> str:
    """Return 'opener', 'closer', or 'transient'. Body is the tag's
    INNER text, stripped + casefolded by caller."""
    if any(sub in tag_body for sub in _TONE_CLOSER_SUBSTRINGS):
        return "closer"
    if tag_body in _TONE_OPENERS:
        return "opener"
    return "transient"


def _check_tone_pairing(
    output_text: str,
    *,
    input_ends_in_blockquote: bool,
) -> tuple[bool, str]:
    """Walk tags in order; auto-close on stacked openers (S2-Pro semantics:
    a new opener overrides the previous one). At chunk end, require any
    open tone be closed UNLESS the input itself ends inside a blockquote
    (legitimate chunk-split mid-quote; the next chunk will continue the
    tonal voice and emit the closer when the quote actually ends).
    """
    open_tone: str | None = None
    for m in _TAG_EXTRACT_RE.finditer(output_text):
        body = m.group(1).strip().casefold()
        if not body:
            # `[ ]` / `[   ]` shapes — Fish s2-pro would render the
            # brackets literally in audio. Round-2 advisor M2.
            return False, "empty tag body in output"
        kind = _classify_tag(body)
        if kind == "opener":
            open_tone = body  # implicit close-then-open per advisor H2
        elif kind == "closer":
            if open_tone is None:
                return False, f"stray tone-closer `[{body}]` with no opener"
            open_tone = None
        # transient: ignore
    if open_tone is not None and not input_ends_in_blockquote:
        return False, f"unclosed tone `[{open_tone}]` at end of chunk"
    return True, ""


# Deterministic paragraph-pause enforcement. The user wants every
# paragraph boundary to carry a beat at minimum, regardless of whether
# Claude's prosody pass decided to tag it. Idempotent: skips paragraphs
# whose tail already ends with a tag that would make an extra pause
# noisy (existing pause/beat/breath OR a tone-closer like
# `[back to narration]` where a stacked beat would feel mechanical).
# Also skips the FINAL non-empty paragraph (no beat needed at EOF).
# Pause tag emitted at paragraph boundaries. tts-tool's chunker parses
# this out, splits the text into Fish-callable segments at each
# occurrence, and inserts a deterministic silence-MP3 of the matching
# duration during stitch. So the AUDIO duration is decided by
# tts-tool's parse_pause_duration map, NOT by Fish.
#
# Mapping (in tts-tool/chunk.py::parse_pause_duration):
#   [short pause]         -> 0.3 s
#   [pause]               -> 0.5 s
#   [long pause]          -> 1.0 s
#   [very long pause]     -> 1.5 s
#   [N second pause]      -> N    s
#
# 1.0 s reads as a clear paragraph beat without stalling. Bump to
# [very long pause] for slower / more deliberate narration.
_PARAGRAPH_PAUSE_TAG = "[long pause]"
_TAG_AT_TAIL_RE = re.compile(r"\[([^\[\]\n]+)\]\s*[^\w]*$")

# Tag-body substrings that mean "no extra pause needed after this".
# These MUST be pause-rendering tags only — anything that produces
# real silence in the audio. Tone closers (`[back to narration]`,
# `[narrator's voice]`, etc.) are NOT in this list anymore: closers
# are tone-state markers, not silence-producing. Empirical bug
# 2026-05-22: paragraph ending in `[back to narration]` bled directly
# into the next paragraph because the closer suppressed the
# paragraph-pause insertion. Closers no longer suppress.
#
# These needles must not overlap any entry in `_TONE_OPENERS` — overlap
# would silently skip pause after an opener (a worse bug). Currently
# safe.
_TAIL_TAG_SKIPS_PAUSE = (
    "pause",
    "beat",
    "breath",
)


def _paragraph_already_paused(paragraph_text: str) -> bool:
    """True if the paragraph's tail ends with a tag whose presence
    obviates appending another `[short pause]`."""
    m = _TAG_AT_TAIL_RE.search(paragraph_text)
    if not m:
        return False
    body = m.group(1).strip().casefold()
    return any(needle in body for needle in _TAIL_TAG_SKIPS_PAUSE)


def enforce_paragraph_pauses(text: str) -> str:
    """Insert `[short pause]` between paragraphs that don't already
    carry a tail tag making one redundant. Idempotent across multiple
    passes.

    The FINAL non-empty paragraph is intentionally skipped — a beat at
    EOF just hangs ~250ms of silence (advisor round-1 H2). Passthrough
    chunks (untagged Markdown) participate normally; the pause is
    deterministic prosody regardless of source.
    """
    paragraphs = text.split("\n\n")
    # Index of the last non-empty paragraph — that's the one to skip.
    last_nonempty = -1
    for i, para in enumerate(paragraphs):
        if para.rstrip():
            last_nonempty = i
    out: list[str] = []
    for i, para in enumerate(paragraphs):
        stripped = para.rstrip()
        if not stripped:
            out.append(para)
            continue
        if i == last_nonempty:
            out.append(para)
            continue
        if _paragraph_already_paused(stripped):
            out.append(para)
        else:
            out.append(f"{stripped} {_PARAGRAPH_PAUSE_TAG}")
    return "\n\n".join(out)


def _input_ends_in_blockquote(text: str) -> bool:
    """True if the last non-blank line of the chunk's input is a
    Markdown blockquote (`>` prefix). Used by `_check_tone_pairing` to
    allow legitimate chunk-split-mid-quote without false-positive on
    unclosed tone."""
    for line in reversed(text.split("\n")):
        if line.strip():
            return line.lstrip().startswith(">")
    return False

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

## Tonal voice tags vs transient cues (IMPORTANT)

S2-Pro tonal voice tags PERSIST forward — `[reading aloud]` keeps the
quoted-voice rendering active for everything that follows until
another tone tag overrides it. The ONLY tonal voice openers you may
use are this CLOSED SET:

    [reading aloud]
    [thoughtfully]
    [softly]
    [firmly]
    [warmly]
    [drily]
    [reflectively]
    [matter-of-factly]

Each one you open MUST be closed before chunk-end. Use exactly:
    [back to narration]

The reset goes immediately AFTER the last word of the tonally-shifted
span, BEFORE the next paragraph break.

If you need to express a TRANSIENT delivery cue (e.g. a single beat
of hesitation, an aside), use natural-language brackets that DON'T
match the closed set above — e.g. `[after a moment]`, `[briefly]`,
`[as an aside]`. These are read as transient inflections, not
persistent voice switches, and don't need a close-tag.

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
   - An em-dash can be a long pause OR a short aside-pause-then-resume.
     Read the sentence and decide. Don't tag every em-dash.
   - A blockquote is usually `[reading aloud]` or `[thoughtfully]`. Tag
     the FIRST sentence of the quote, not every line, AND close it with
     `[back to narration]` after the last sentence of the quote.

6. **EMPHASIS ON CONCLUSIONS — be more generous here than elsewhere.**
   When prose builds toward a rhetorical conclusion, tag the operative
   word or short phrase with `[emphasis]`. A human narrator reading
   aloud will stress these naturally; the default cadence won't, so
   the tag is doing real work. Examples (operative span in CAPS, you
   emit `[emphasis] span` in the output):
     "However, the truth is we can be happy NOW." -> emphasize "now"
     "We will always experience unhappiness IF WE BELIEVE IN IT." ->
        emphasize "if we believe in it"
     "The Option Method addresses TOTAL happiness." -> emphasize "total"
     "There is NO reason why you can't be." -> emphasize "no"
   Look for: final pivot words (now, never, only, must, cannot, no),
   conditional clauses that flip meaning (if/unless/until), the LAST
   noun phrase of an aphorism, contrast words after "however / but /
   yet / instead" when the sentence resolves to the speaker's claim.
   You can use [emphasis] more often than tone tags — 1-2 per paragraph
   in argumentative prose is normal. Skip emphasis only when the
   sentence is purely descriptive.

7. **Bold/italic markdown is NOT a signal**. The deterministic
   pre-process strips `**bold**` / `*italic*` before you see the text,
   because most markdown emphasis in articles marks technical terms or
   foreign words (not rhetorical stress). Decide emphasis from PROSE
   MEANING, not from removed-syntax cues.

## Example

INPUT:
# Cats Sleep A Lot

Cats sleep up to **sixteen** hours a day. The exact number depends on
age, diet, and the quality of available sunbeams.

> Older cats sleep more — kittens, surprisingly, also sleep a lot.

That's why their owners install heated beds.

OUTPUT:
Cats Sleep A Lot [long pause] Cats sleep up to [emphasis] sixteen \
hours a day. The exact number depends on age, diet, and the quality \
of available sunbeams. [reading aloud] Older cats sleep more [short \
pause] kittens, surprisingly, also sleep a lot. [back to narration] \
That's why their owners install heated beds.

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


def _validate(
    input_text: str,
    output_text: str,
    *,
    strict_tones: bool = False,
) -> tuple[bool, str]:
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

    `strict_tones=True` adds the tone-pairing check: every opener from
    the closed set in `_TONE_OPENERS` must be closed by chunk-end
    (substring match against `_TONE_CLOSER_SUBSTRINGS`). Ships
    opt-in so the failure rate can be measured before flipping default
    (per advisor M3).
    """
    if strict_tones:
        ok, reason = _check_tone_pairing(
            output_text,
            input_ends_in_blockquote=_input_ends_in_blockquote(input_text),
        )
        if not ok:
            return False, reason
    # strip_tags BOTH sides: bracket-content is TAGS, not prose. The
    # deterministic pre-process emits tags like `[firmly]` around
    # headings — those words are markup, not narrated prose. Word-drift
    # check compares spoken-word streams, so strip tags on both sides
    # to compare apples-to-apples.
    expected_words = _word_tokens(strip_tags(input_text))
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
    strict_tones: bool = False,
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

        ok, reason = _validate(chunk_text, output, strict_tones=strict_tones)
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
