"""Tests for audio_chunk.py — ffmpeg silencedetect primitives + aligned
chunk construction.

The ffmpeg invocations are mocked end-to-end: tests run on hosts that
don't have ffmpeg, and even on hosts that do we don't want to depend on
a real audio file being present. The orchestration logic (silence
parsing, target-boundary snapping, proportional text distribution) is
all pure Python on top of those primitives and is the part that
actually matters.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from prose_decorate.audio_chunk import (
    AudioWindow,
    align_text_to_audio_chunks,
    audio_size_bytes,
    parse_silencedetect_output,
    partition_audio_at_silences,
    snap_to_nearest_silence_midpoint,
)


# ---------- parse_silencedetect_output ----------

# A canonical chunk of `ffmpeg ... -af silencedetect` stderr output.
# silencedetect emits lines in pairs: `silence_start: <s>` and
# `silence_end: <s> | silence_duration: <s>`.
_SAMPLE_SILENCEDETECT = """\
ffmpeg version 6.0 ...
[silencedetect @ 0x55] silence_start: 12.3
[silencedetect @ 0x55] silence_end: 13.1 | silence_duration: 0.8
[silencedetect @ 0x55] silence_start: 45.6
[silencedetect @ 0x55] silence_end: 46.4 | silence_duration: 0.8
[silencedetect @ 0x55] silence_start: 90.0
[silencedetect @ 0x55] silence_end: 91.2 | silence_duration: 1.2
size=  ... time=00:01:30.00 ...
"""


def test_parse_silencedetect_extracts_pairs():
    pairs = parse_silencedetect_output(_SAMPLE_SILENCEDETECT)
    assert pairs == [(12.3, 13.1), (45.6, 46.4), (90.0, 91.2)]


def test_parse_silencedetect_handles_trailing_unclosed_start():
    """Real-world: silencedetect may emit a final `silence_start` without
    a paired `silence_end` if the audio ends in silence. Skip that
    incomplete pair rather than crashing — the file ended in silence;
    there's no useful split there anyway."""
    out = (
        "[silencedetect @ 0x55] silence_start: 1.0\n"
        "[silencedetect @ 0x55] silence_end: 2.0 | silence_duration: 1.0\n"
        "[silencedetect @ 0x55] silence_start: 100.0\n"
    )
    pairs = parse_silencedetect_output(out)
    assert pairs == [(1.0, 2.0)]


def test_parse_silencedetect_empty_returns_empty():
    assert parse_silencedetect_output("") == []
    assert parse_silencedetect_output("no silences found here") == []


# ---------- snap_to_nearest_silence_midpoint ----------

def test_snap_picks_closest_silence_midpoint():
    silences = [(10.0, 11.0), (50.0, 51.0), (100.0, 101.0)]
    # Target 49s -> closest midpoint is 50.5
    assert snap_to_nearest_silence_midpoint(49.0, silences) == pytest.approx(50.5)


def test_snap_falls_back_to_target_when_no_silences():
    assert snap_to_nearest_silence_midpoint(30.0, []) == 30.0


def test_snap_picks_first_when_target_before_all():
    silences = [(50.0, 51.0), (100.0, 101.0)]
    # Target 5s -> first silence midpoint 50.5
    assert snap_to_nearest_silence_midpoint(5.0, silences) == pytest.approx(50.5)


# ---------- partition_audio_at_silences ----------

def test_partition_produces_windows_summing_to_duration():
    silences = [(50.0, 51.0), (100.0, 101.0), (150.0, 151.0)]
    windows = partition_audio_at_silences(
        silences, total_duration=200.0, target_chunk_seconds=60.0,
    )
    # Windows are contiguous and cover the full duration.
    assert windows[0].start == 0.0
    assert windows[-1].end == pytest.approx(200.0)
    for a, b in zip(windows, windows[1:], strict=False):
        assert a.end == b.start
    # Each window roughly the target size (snapped to a silence).
    for w in windows[:-1]:
        # Should be within ~1.5x target (snap can extend; last is the
        # remainder so it may be smaller).
        assert (w.end - w.start) <= 1.5 * 60.0


def test_partition_returns_single_window_for_short_audio():
    """Audio shorter than target -> one window covering everything."""
    windows = partition_audio_at_silences(
        [], total_duration=30.0, target_chunk_seconds=300.0,
    )
    assert len(windows) == 1
    assert windows[0] == AudioWindow(start=0.0, end=30.0)


def test_partition_no_silences_falls_back_to_uniform():
    """No silences at all but audio > target -> fall back to uniform
    split at target boundaries (better than producing a single
    over-budget window)."""
    windows = partition_audio_at_silences(
        [], total_duration=300.0, target_chunk_seconds=100.0,
    )
    assert len(windows) == 3
    assert windows[0].start == 0.0
    assert windows[-1].end == 300.0


# ---------- align_text_to_audio_chunks ----------

def test_align_text_distributes_proportionally_by_word_count():
    """Text 'a b c d e f g h' (8 words) across 2 equal windows ->
    4 words per window."""
    windows = [AudioWindow(0.0, 60.0), AudioWindow(60.0, 120.0)]
    text = "a b c d e f g h"
    aligned = align_text_to_audio_chunks(text, windows)
    assert len(aligned) == 2
    assert len(aligned[0].split()) == 4
    assert len(aligned[1].split()) == 4
    # Reassembly preserves the word stream.
    assert " ".join(aligned).split() == text.split()


def test_align_text_proportional_to_window_durations():
    """Window 1 is twice as long as window 2 -> twice the text."""
    windows = [AudioWindow(0.0, 100.0), AudioWindow(100.0, 150.0)]
    text = " ".join([f"w{i}" for i in range(30)])  # 30 words
    aligned = align_text_to_audio_chunks(text, windows)
    assert len(aligned) == 2
    # 100/150 = 2/3 of text in first window -> ~20 words, ~10 in second.
    assert 18 <= len(aligned[0].split()) <= 22
    assert 8 <= len(aligned[1].split()) <= 12


def test_align_text_single_window_returns_full_text():
    windows = [AudioWindow(0.0, 60.0)]
    text = "Hello world. This is a test."
    aligned = align_text_to_audio_chunks(text, windows)
    assert aligned == [text]


def test_align_text_empty_text_produces_empty_chunks():
    """Edge: empty transcript with audio still produces matching
    empty chunks rather than crashing on division by zero."""
    windows = [AudioWindow(0.0, 60.0), AudioWindow(60.0, 120.0)]
    aligned = align_text_to_audio_chunks("", windows)
    assert aligned == ["", ""]


# ---------- audio_size_bytes ----------

def test_audio_size_bytes_reads_file_size(tmp_path: Path):
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"\xff\xfb" + b"X" * 1024)
    assert audio_size_bytes(audio) == 1026
