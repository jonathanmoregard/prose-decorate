"""Deterministic Markdown -> plain prose pre-process.

Handles every element the LLM SHOULDN'T have to judge — code blocks,
images, tables, footnotes, links — by translating them to plain-text
tokens or stripping them outright. Leaves inline emphasis
(`**bold**`, `*italic*`), blockquotes, em-dashes, and ellipses in
place so the LLM-decoration stage can make the prosody calls.

Why deterministic here: these elements have one obvious correct
spoken form (or no spoken form at all) and don't need contextual
judgment.
"""
from __future__ import annotations

import re

# Order matters: code blocks first (fenced runs may contain anything),
# then images, then links, then inline code, then tables/footnotes,
# then header stripping.

_FENCED_CODE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)
_INDENTED_CODE_BLOCK = re.compile(
    r"(?m)(?:^(?:    |\t).*\n?)+"
)
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
# Bare URLs in prose. Substack writers paste raw URLs all the time (`Here
# is a link: https://choosehappiness.net/about-bruce/`). TTS reads them
# verbatim — "h-t-t-p-s colon slash slash..." — which is unlistenable.
# Replace with a speakable token. Match permissive scheme + host + path
# chars; stop at whitespace, closing-paren, or trailing punct like `,` /
# `.` (a sentence-final period after a URL is ambiguous; we accept the
# rare false positive over emitting "dot html" as a sentence terminator).
_BARE_URL = re.compile(r"https?://[^\s<>()\[\]]+[^\s<>()\[\].,;:!?]")
_REFERENCE_LINK = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
# `[label]: https://...` reference-link definition rows. Multiline so we
# match each one independently; pattern requires the leading `[label]:`
# shape on its own line so it doesn't catch in-prose footnotes (those
# carry `^` after the bracket and are handled by _FOOTNOTE_DEF).
_REFERENCE_LINK_DEF = re.compile(r"(?m)^\[[^\]^]+\]:\s+\S.*$")
_INLINE_CODE = re.compile(r"`[^`\n]+`")
# GFM-ish table: a line whose stripped form is `| ... |` with `|`
# separators; covers header rows, alignment rows, and body rows.
_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_ALIGN = re.compile(r"^\s*\|?[\s:-]+\|[\s:-|]+\|?\s*$")
_FOOTNOTE_DEF = re.compile(r"^\[\^[^\]]+\]:.*$", re.MULTILINE)
_FOOTNOTE_REF_INLINE = re.compile(r"\[\^[^\]]+\]")
_HEADING_HASH = re.compile(r"^(#{1,6})\s+", re.MULTILINE)
_HORIZONTAL_RULE = re.compile(r"^[ \t]*(?:-{3,}|\*{3,}|_{3,})[ \t]*$", re.MULTILINE)


def strip_markdown(text: str) -> str:
    """Apply the v0.1 markdown handling matrix. Returns plain prose
    suitable for chunking + LLM decoration. Idempotent on already-plain
    text (no markdown -> no change beyond whitespace normalization)."""
    # Fenced code blocks -> [code omitted] line. Process before inline-code
    # regex so triple-backtick content can't be mis-matched as inline.
    text = _FENCED_CODE.sub("[code omitted]", text)
    # Images: keep alt text only (often more useful than dropping silently;
    # alt of "" yields "" which collapses harmlessly via blank-line dedup).
    text = _IMAGE.sub(r"\1", text)
    # Inline links: keep link text only.
    text = _LINK.sub(r"\1", text)
    text = _REFERENCE_LINK.sub(r"\1", text)
    # Drop bottom-of-document reference-link definitions.
    text = _REFERENCE_LINK_DEF.sub("", text)
    # Bare URLs in prose -> speakable token. Done AFTER markdown-link
    # rewrites so a `[text](url)` pair has already lost its URL by now.
    text = _BARE_URL.sub("[link]", text)
    # Inline code -> [code] token (TTS would otherwise read literal punct).
    text = _INLINE_CODE.sub("[code]", text)
    # Tables -> single [table omitted] line per table-run. Strategy:
    # walk lines, replace runs of table-shaped lines with one token.
    text = _strip_tables(text)
    # Footnote definitions: drop entirely (we already dropped footnote
    # references upstream in substack-url-tool's clean_substack; this is
    # belt-and-braces for non-Substack inputs).
    text = _FOOTNOTE_DEF.sub("", text)
    text = _FOOTNOTE_REF_INLINE.sub("", text)
    # Heading hashes: drop leading `#`s but keep the heading text as its
    # own paragraph (blank-line surrounded). The LLM uses the paragraph
    # boundary to decide whether to insert a `[long pause]`.
    text = _HEADING_HASH.sub("", text)
    # Horizontal rules -> blank line.
    text = _HORIZONTAL_RULE.sub("", text)
    # Collapse 3+ blank lines to 2 (one paragraph break).
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_tables(text: str) -> str:
    """Replace runs of table lines (incl. the |---|---| alignment row) with
    a single [table omitted] line."""
    lines = text.split("\n")
    out: list[str] = []
    in_table = False
    for line in lines:
        is_table = bool(_TABLE_LINE.match(line) or _TABLE_ALIGN.match(line))
        if is_table:
            if not in_table:
                out.append("[table omitted]")
                in_table = True
            continue
        in_table = False
        out.append(line)
    return "\n".join(out)


# Smart-quote / dash → ASCII fold. Sonnet at temperature=0 still
# normalizes typography in its output (curly apostrophes -> straight,
# en/em dashes -> hyphens occasionally), so the drift guard has to
# fold input the same way to compare apples-to-apples.
_SMART_FOLD = {
    "\u2018": "'", "\u2019": "'",   # ‘ ’
    "\u201c": '"', "\u201d": '"',   # “ ”
    "\u2013": "-", "\u2014": "-",   # – —
    "\u2026": "...",                 # …
    "\u00a0": " ",                   # non-breaking space
}

# Markdown emphasis markers that the LLM is INSTRUCTED to drop from
# output (the system prompt tells it to decide whether to tag bold/
# italic, then drop the syntax). The drift guard must therefore strip
# them from BOTH sides of the equality check.
#
# Broad pattern: strip ALL `*` and `_` runs. This deliberately also
# nukes underscores in identifiers and literal asterisks — that's
# safe for COMPARISON purposes because both sides are stripped the
# same way; the drift guard only needs to detect WORD changes, not
# punctuation/syntax differences. A previous narrow regex required
# word-char adjacency on at least one side, which missed asymmetric
# emphasis like `*Summary:*` (trailing `*` preceded by `:`, not by a
# word char) — caught when a real article tripped it (2026-05-21).
_EMPHASIS_RE = re.compile(r"[*_]+")


def _fold_typography(text: str) -> str:
    for src, dst in _SMART_FOLD.items():
        text = text.replace(src, dst)
    return text


def normalize_whitespace(text: str) -> str:
    """Canonical form for the strip_tags(output) == normalize_whitespace(input)
    safety check in decorate.py.

    Whitespace:
    - collapse `[ \\t]+` runs to single space
    - strip leading AND trailing whitespace per line (so a stray
      leading space the model emitted on a quote line is tolerated)
    - collapse SINGLE newlines (soft line breaks) to single space —
      Markdown soft-wrap semantics + Sonnet's tendency to join them
      mean strict newline preservation would false-positive the drift
      guard on every chunk
    - preserve `\\n\\n` as paragraph boundary
    - trim outer whitespace

    Typography (added 2026-05-21 after a real article tripped the
    guard on curly apostrophes):
    - fold smart quotes / em-dash / ellipsis to ASCII equivalents
    - strip markdown emphasis runs (`*` / `_`) adjacent to word chars

    Explicit non-changes: NO case change, NO Unicode NFC across the
    whole string, NO general punctuation strip — the guard is meant to
    catch the LLM CHANGING WORDS, not stylistic typography that the
    system prompt explicitly invites it to normalize.
    """
    text = _fold_typography(text)
    text = _EMPHASIS_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [ln.strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    # Collapse 3+ newlines to 2 (paragraph boundary) FIRST so we can
    # then collapse remaining single newlines without touching the
    # paragraph boundary markers.
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Mark paragraph boundaries with a placeholder, collapse soft
    # newlines to spaces, then restore.
    text = text.replace("\n\n", "\x00PARA\x00")
    text = re.sub(r"\s*\n\s*", " ", text)
    text = text.replace("\x00PARA\x00", "\n\n")
    # Re-collapse any spaces the soft-newline pass introduced next to
    # existing single-space runs.
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


_TAG_RE = re.compile(r"\[[^\[\]\n]+\]")


def strip_tags(decorated: str) -> str:
    """Remove all `[bracket-content]` Fish s2-pro tags from decorated text.

    Preserved tokens the deterministic pre-process emits — `[code]`,
    `[code omitted]`, `[table omitted]` — are also bracket-shaped, so
    the regex strips them too. That's correct: the input fed to the
    LLM already contains those tokens, so they survive round-trip
    via the strip_tags(output) == normalize_whitespace(input) check.
    """
    return _TAG_RE.sub("", decorated)
