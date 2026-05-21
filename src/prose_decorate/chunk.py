"""Paragraph-boundary chunker with sentence-fallback for oversize paragraphs.

# Keep in sync with tts-tool/src/tts_tool/chunk.py — same greedy-pack
# semantics (never split a sentence, prefer paragraph boundaries when
# chunk is already above the soft threshold). When this duplication
# starts to bite, extract to a shared library. For v0.1 the cost of
# the second copy is ~80 LOC and the cost of taking a dependency on
# tts-tool just for chunking would be larger.

Lightweight sentence splitter — regex on `[.!?](\\s|$)` boundaries with
abbreviation guards. This is "good enough" for English prose article
chunking; tts-tool's spaCy en_core_web_sm pipeline is more accurate but
overkill here (sentence-split is only the fallback path for paragraphs
that exceed the chunk budget; almost every paragraph fits whole).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

TARGET_CHARS = 4000
SOFT_CLOSE_CHARS = 2500
PREV_CONTEXT_CHARS = 400


@dataclass(frozen=True)
class Chunk:
    text: str
    index: int
    prev_context: str  # "" for first chunk


# Don't split on a `.` that's part of a known abbreviation or initial.
# Conservative list; better to under-split than over-split (LLM tolerates
# slightly long chunks; tts-tool prefers them too).
_ABBREV = (
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr",
    "St", "Mt", "Ft", "vs", "etc", "e.g", "i.e", "cf",
    "Inc", "Ltd", "Co", "Corp",
)
_ABBREV_RE = re.compile(
    r"\b(?:" + "|".join(_ABBREV) + r")\.$"
)


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]


def split_sentences(paragraph: str) -> list[str]:
    """Regex sentence splitter. Splits on `.`, `!`, or `?` followed by
    whitespace, but holds back when the preceding token is a known
    abbreviation."""
    parts = re.split(r"(?<=[.!?])\s+", paragraph)
    out: list[str] = []
    buf = ""
    for part in parts:
        candidate = (buf + " " + part).strip() if buf else part
        if _ABBREV_RE.search(part):
            buf = candidate
        else:
            out.append(candidate)
            buf = ""
    if buf:
        out.append(buf)
    return [s for s in (p.strip() for p in out) if s]


def _last_paragraph(text: str) -> str:
    paras = split_paragraphs(text)
    if not paras:
        return ""
    last = paras[-1]
    if len(last) > PREV_CONTEXT_CHARS:
        return "…" + last[-PREV_CONTEXT_CHARS:]
    return last


def chunk_markdown(
    text: str,
    *,
    target: int = TARGET_CHARS,
    soft: int = SOFT_CLOSE_CHARS,
) -> list[Chunk]:
    """Paragraph-boundary greedy pack. Sentence-boundary fallback when a
    single paragraph exceeds `target`."""
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return []

    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_len = 0

    def emit(prev_full_text: str) -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        chunks.append(
            Chunk(
                text="\n\n".join(buf),
                index=len(chunks),
                prev_context=_last_paragraph(prev_full_text),
            )
        )
        buf = []
        buf_len = 0

    prev_emitted = ""
    for para in paragraphs:
        plen = len(para)

        # Oversize paragraph: emit current buf first, then sentence-split
        # this paragraph into chunks of its own.
        if plen > target:
            emit(prev_emitted)
            prev_emitted = "\n\n".join(chunks[-1].text.split("\n\n")) if chunks else ""
            sent_chunks = _split_oversize(para, target, soft, len(chunks), prev_emitted)
            for sc in sent_chunks:
                chunks.append(sc)
                prev_emitted = sc.text
            continue

        # Fit-in-current-buf?
        sep = 2 if buf else 0  # "\n\n" between paragraphs
        if buf and buf_len + sep + plen > target:
            emit(prev_emitted)
            prev_emitted = chunks[-1].text if chunks else ""

        buf.append(para)
        buf_len += sep + plen

        # Soft close: once we're above soft threshold, end at this para
        # boundary even if the next would fit. Keeps chunks roughly the
        # same size and gives the LLM a clean stopping point.
        if buf_len >= soft:
            emit(prev_emitted)
            prev_emitted = chunks[-1].text if chunks else ""

    emit(prev_emitted)
    return chunks


def _split_oversize(
    paragraph: str,
    target: int,
    soft: int,
    start_index: int,
    initial_prev: str,
) -> list[Chunk]:
    sentences = split_sentences(paragraph)
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_len = 0
    prev = initial_prev
    for sent in sentences:
        sep = 1 if buf else 0
        if buf and buf_len + sep + len(sent) > target:
            chunks.append(
                Chunk(
                    text=" ".join(buf),
                    index=start_index + len(chunks),
                    prev_context=_last_paragraph(prev),
                )
            )
            prev = " ".join(buf)
            buf = []
            buf_len = 0
        buf.append(sent)
        buf_len += sep + len(sent)
    if buf:
        chunks.append(
            Chunk(
                text=" ".join(buf),
                index=start_index + len(chunks),
                prev_context=_last_paragraph(prev),
            )
        )
    return chunks
