"""Audio-aligned chunking primitives.

The Claude path chunks at paragraph boundaries with a soft target of
~4000 chars. The audio path adds one extra constraint: each transcript
chunk must come paired with the audio segment that contains its
narration, so Gemini can ground tag decisions on real delivery.

For audio that fits a single Gemini inline-data request (≤~18MB),
the CLI sends the whole audio with each text chunk and we never need
the silence-detection path. This module is the LONG-AUDIO path —
ffmpeg silencedetect for natural boundary discovery, then text
distribution proportional to window durations.

ffmpeg is invoked via subprocess from the CLI, NOT from this module.
Pure functions here are easy to unit-test; the orchestration glue
that runs ffmpeg lives next to its only caller. Same separation we
use everywhere else (parse vs run).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioWindow:
    """A `[start, end)` time window in seconds within an audio file."""
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


# Lines like `[silencedetect @ 0x55] silence_start: 12.3`
_SILENCE_START_RE = re.compile(
    r"silence_start:\s*([0-9]+(?:\.[0-9]+)?)"
)
# `[silencedetect @ 0x55] silence_end: 13.1 | silence_duration: 0.8`
_SILENCE_END_RE = re.compile(
    r"silence_end:\s*([0-9]+(?:\.[0-9]+)?)"
)


def parse_silencedetect_output(stderr_text: str) -> list[tuple[float, float]]:
    """Parse ffmpeg silencedetect stderr into `(start, end)` pairs.

    silencedetect emits start/end on separate lines and may emit a
    trailing unpaired `silence_start` if the audio ends in silence —
    we skip that incomplete pair (a split at the very end of the file
    isn't a useful boundary anyway).
    """
    starts: list[float] = []
    ends: list[float] = []
    for line in stderr_text.splitlines():
        m = _SILENCE_START_RE.search(line)
        if m:
            starts.append(float(m.group(1)))
            continue
        m = _SILENCE_END_RE.search(line)
        if m:
            ends.append(float(m.group(1)))
    # Pair up; drop trailing unpaired start.
    n = min(len(starts), len(ends))
    return [(starts[i], ends[i]) for i in range(n)]


def snap_to_nearest_silence_midpoint(
    target_seconds: float,
    silences: list[tuple[float, float]],
) -> float:
    """Pick the silence whose midpoint is closest to `target_seconds`
    and return that midpoint. Falls back to `target_seconds` when
    `silences` is empty (caller must handle the uniform-split path)."""
    if not silences:
        return target_seconds
    best_mid = None
    best_dist = float("inf")
    for s, e in silences:
        mid = (s + e) / 2.0
        d = abs(mid - target_seconds)
        if d < best_dist:
            best_dist = d
            best_mid = mid
    assert best_mid is not None  # silences non-empty
    return best_mid


def partition_audio_at_silences(
    silences: list[tuple[float, float]],
    *,
    total_duration: float,
    target_chunk_seconds: float,
) -> list[AudioWindow]:
    """Split [0, total_duration) into contiguous windows of approximately
    `target_chunk_seconds`, with each split point snapped to the
    nearest silence midpoint.

    - Audio shorter than target -> single window.
    - No silences at all but audio > target -> uniform split (better
      than emitting one over-budget window — the caller still needs
      to fit each window under the Gemini inline cap).

    Each silence is consumed at most once: after we snap a boundary to
    a silence we drop it from the candidate pool so the next boundary
    can't pick the same silence and produce a zero-length window.
    """
    if total_duration <= target_chunk_seconds:
        return [AudioWindow(start=0.0, end=total_duration)]

    # Boundaries we'd ideally cut at: target, 2*target, ...
    raw_boundaries = []
    t = target_chunk_seconds
    while t < total_duration:
        raw_boundaries.append(t)
        t += target_chunk_seconds

    if not silences:
        # Uniform split fallback.
        boundaries = raw_boundaries
    else:
        candidates = list(silences)
        boundaries = []
        for rb in raw_boundaries:
            if not candidates:
                boundaries.append(rb)
                continue
            best_idx = min(
                range(len(candidates)),
                key=lambda i: abs(((candidates[i][0] + candidates[i][1]) / 2.0) - rb),
            )
            s, e = candidates.pop(best_idx)
            boundaries.append((s + e) / 2.0)

    # Ensure strictly increasing — if two raw targets snapped to the
    # same silence (shouldn't happen now we pop, but defensive) drop
    # duplicates. Then keep only those strictly inside (0, duration).
    cleaned: list[float] = []
    for b in boundaries:
        if 0 < b < total_duration and (not cleaned or b > cleaned[-1]):
            cleaned.append(b)

    windows: list[AudioWindow] = []
    prev = 0.0
    for b in cleaned:
        windows.append(AudioWindow(start=prev, end=b))
        prev = b
    windows.append(AudioWindow(start=prev, end=total_duration))
    return windows


def align_text_to_audio_chunks(
    text: str,
    windows: list[AudioWindow],
) -> list[str]:
    """Distribute `text` across `windows` proportionally to each
    window's duration, splitting at word boundaries.

    Round-down on cumulative-word-boundaries: window i gets words
    `[wstart_i, wstart_i+1)` where `wstart_i` is the cumulative
    proportional word index. The last window absorbs any rounding
    remainder so the concatenated word stream matches the input
    exactly (the word-drift validator demands it).

    Edge cases:
    - empty text -> list of empty strings (same length as windows)
    - single window -> the full text in one entry, unchanged
    """
    if not windows:
        return []
    if len(windows) == 1:
        return [text]
    words = text.split()
    if not words:
        return ["" for _ in windows]

    total_duration = sum(w.duration for w in windows)
    if total_duration <= 0:
        # Pathological: all windows zero-length. Spread uniformly.
        return _uniform_word_split(words, len(windows))

    boundaries: list[int] = [0]
    cum = 0.0
    for w in windows[:-1]:
        cum += w.duration
        idx = int(round((cum / total_duration) * len(words)))
        # Monotone non-decreasing, never exceed total
        idx = max(boundaries[-1], min(idx, len(words)))
        boundaries.append(idx)
    boundaries.append(len(words))

    return [
        " ".join(words[boundaries[i]:boundaries[i + 1]])
        for i in range(len(windows))
    ]


def _uniform_word_split(words: list[str], n: int) -> list[str]:
    if n <= 0:
        return []
    base = len(words) // n
    extra = len(words) % n
    out: list[str] = []
    i = 0
    for k in range(n):
        sz = base + (1 if k < extra else 0)
        out.append(" ".join(words[i:i + sz]))
        i += sz
    return out


def audio_size_bytes(path: Path | str | os.PathLike[str]) -> int:
    """File size in bytes — used by the CLI to decide between the
    inline-single-shot path and the silence-chunked path."""
    return os.path.getsize(path)
