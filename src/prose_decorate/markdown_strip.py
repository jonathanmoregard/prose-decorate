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


def normalize_whitespace(text: str) -> str:
    """Canonical form for the strip_tags(output) == normalize_whitespace(input)
    safety check in decorate.py. Defined narrowly:

    - collapse `[ \\t]+` runs to single space
    - collapse SINGLE newlines (soft line breaks) to single space —
      Markdown soft-wrap semantics + Sonnet's tendency to join them
      mean strict newline preservation would false-positive the drift
      guard on every chunk
    - preserve `\\n\\n` as paragraph boundary
    - strip trailing whitespace per line
    - trim outer whitespace

    Explicit non-changes: NO case change, NO Unicode NFC, NO punctuation
    strip.
    """
    text = re.sub(r"[ \t]+", " ", text)
    lines = [ln.rstrip() for ln in text.split("\n")]
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
