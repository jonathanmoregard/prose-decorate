"""prose-decorate entrypoint. Markdown -> Fish s2-pro-tagged plain text."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import audio as audio_mod
from . import cache, decorate
from .chunk import Chunk, chunk_markdown
from .markdown_strip import strip_markdown


EXIT_OK = 0
EXIT_INVALID = 1
EXIT_EMPTY = 2
EXIT_TTY = 3
EXIT_PARTIAL_PASSTHROUGH = 10
EXIT_FULL_PASSTHROUGH = 20


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="prose-decorate",
        description=(
            "Markdown on stdin -> Fish s2-pro prosody-tagged plain text on "
            "stdout. Sits between substack-url-tool --format markdown and "
            "tts-tool."
        ),
    )
    p.add_argument("-i", "--input", type=Path, default=None,
                   help="Read Markdown from FILE instead of stdin.")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Write decorated text to FILE instead of stdout.")
    p.add_argument("--no-cache", action="store_true",
                   help="Bypass the per-chunk decoration cache.")
    p.add_argument("--no-llm", action="store_true",
                   help="Skip the LLM entirely; emit markdown-stripped text "
                        "(passthrough). Useful for offline / debug runs.")
    p.add_argument("--debug", type=Path, default=None, metavar="DIR",
                   help="Write per-chunk {input,output,status,reason} JSON "
                        "sidecars to DIR for introspection.")
    p.add_argument("--strict-tones", action="store_true",
                   help="Reject chunks whose tonal voice tags (`[reading aloud]`, "
                        "`[thoughtfully]`, etc.) aren't closed with a "
                        "`[back to narration]` (or substring variant) by "
                        "chunk-end. Fails -> passthrough that chunk. Off by "
                        "default while passthrough rate is being measured.")
    p.add_argument("--register", type=str, default="", metavar="HINT",
                   help="Free-form natural-language constraint biasing all "
                        "tag choices toward a register. Example: "
                        "'calm, meandering, sleepy bedtime narration'. "
                        "Prepended to the system prompt; cached separately.")
    p.add_argument("--audio", type=Path, default=None, metavar="FILE",
                   help="Multimodal audio-grounded decoration. Reads the "
                        "audio FILE (mp3/wav/ogg/flac) and routes each "
                        "chunk to Gemini 2.5 Pro alongside the audio so "
                        "tag decisions are based on what is HEARD, not "
                        "inferred from prose. Requires GEMINI_API_KEY "
                        "(or GEMINI_API_KEY_FILE) and the `google-genai` "
                        "Python package. Cached separately from text-only "
                        "runs (audio bytes hashed into the cache key).")
    return p


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _read_input(input_path: Path | None) -> str:
    if input_path is not None:
        return input_path.read_text(encoding="utf-8", errors="replace")
    return sys.stdin.buffer.read().decode("utf-8", errors="replace")


def _write_output(text: str, output: Path | None) -> None:
    if output is not None:
        output.write_text(text, encoding="utf-8")
        return
    sys.stdout.write(text)
    sys.stdout.flush()


def _model() -> str:
    return os.environ.get("PROSE_DECORATE_MODEL", decorate.DEFAULT_MODEL).strip() \
        or decorate.DEFAULT_MODEL


def _audio_model() -> str:
    return os.environ.get(
        "PROSE_DECORATE_AUDIO_MODEL", audio_mod.AUDIO_DEFAULT_MODEL
    ).strip() or audio_mod.AUDIO_DEFAULT_MODEL


# Used for the Gemini cache-key "api_version" slot. Audio runs have no
# Anthropic-style date-pinned API header, so we stamp a synthetic value
# that bumps when the audio path's behaviour changes — currently "v1".
# Bump this string when the audio prompt or call shape changes in a
# way that invalidates prior cache entries.
_AUDIO_API_VERSION = "gemini-audio-v1"


# MIME types Gemini accepts for inline audio. Mapping from file
# extension to the canonical mime string we'll hand the SDK. Unknown
# extensions fall back to audio/mp3 with a warning — most podcast
# content is MP3 anyway.
_AUDIO_MIME_BY_EXT = {
    ".mp3": "audio/mp3",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".aac": "audio/aac",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
}


def _detect_audio_mime(path: Path) -> str:
    return _AUDIO_MIME_BY_EXT.get(path.suffix.lower(), audio_mod.DEFAULT_AUDIO_MIME)


def _cache_root() -> str | None:
    return os.environ.get("PROSE_DECORATE_CACHE_DIR") or None


def _emit_debug(
    debug_dir: Path, idx: int, *, input_text: str, output_text: str,
    status: str, reason: str,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "chunk_index": idx,
        "input": input_text,
        "output": output_text,
        "status": status,
        "reason": reason,
    }
    (debug_dir / f"chunk-{idx:04d}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _process_audio(
    chunks: list[Chunk],
    *,
    audio_path: Path,
    no_cache: bool,
    strict_tones: bool,
    debug_dir: Path | None,
    register: str = "",
) -> tuple[list[str], int, int]:
    """Gemini multimodal path. Reads the audio once and sends it
    inline with every chunk. For v0.1 we don't silence-split the
    audio per-chunk — the Gemini 2.5 Pro context window comfortably
    fits multi-hour audio at standard bitrates, so paying to upload
    the same clip with each chunk request is the simplest correct
    thing that keeps the per-chunk cache + validation infrastructure
    unchanged. A follow-up that segment-aligns audio with text chunks
    via audio_chunk.py is the next step when articles get long
    enough to actually push past the inline cap."""
    audio_bytes = audio_path.read_bytes()
    audio_hash = audio_mod.audio_hash_for(audio_bytes)
    audio_mime = _detect_audio_mime(audio_path)

    try:
        api_key = audio_mod.read_gemini_api_key()
    except audio_mod.MissingGeminiAPIKey as e:
        _log(f"warning: {e}; falling back to passthrough")
        for c in chunks:
            if debug_dir:
                _emit_debug(
                    debug_dir, c.index,
                    input_text=c.text, output_text=c.text,
                    status="passthrough", reason=str(e),
                )
        return ([c.text for c in chunks], 0, len(chunks))

    try:
        client = audio_mod.make_client(api_key)
    except audio_mod.MissingGoogleGenAI as e:
        _log(f"warning: {e}; falling back to passthrough")
        for c in chunks:
            if debug_dir:
                _emit_debug(
                    debug_dir, c.index,
                    input_text=c.text, output_text=c.text,
                    status="passthrough", reason=str(e),
                )
        return ([c.text for c in chunks], 0, len(chunks))

    cdir = cache.chunks_dir(_cache_root())
    # Reuse the text prompt-template hash so register / prompt edits to
    # the SHARED prompt still invalidate the audio cache. The audio
    # addendum lives in audio.py; we stamp _AUDIO_API_VERSION to bump
    # the audio cache when the addendum itself changes.
    prompt_hash = decorate.prompt_template_hash()
    model = _audio_model()

    pieces: list[str] = []
    decorated = 0
    passthrough = 0

    soft_cap = audio_mod.INLINE_AUDIO_SOFT_CAP_BYTES
    if len(audio_bytes) > soft_cap:
        _log(
            f"warning: audio is {len(audio_bytes):,} bytes (> {soft_cap:,} "
            "inline-data soft cap). v0.1 still sends it whole; large "
            "audio may fail Gemini's per-request limit. Follow-up: "
            "silence-chunk via audio_chunk.partition_audio_at_silences."
        )

    for c in chunks:
        cache_key = cache.key_for(
            chunk_text=c.text,
            prev_context=c.prev_context,
            prompt_template_hash=prompt_hash,
            model=model,
            api_version=_AUDIO_API_VERSION,
            register=register,
            audio_hash=audio_hash,
        )
        if not no_cache:
            hit = cache.get(cdir, cache_key)
            if hit is not None:
                _log(f"chunk {c.index + 1}/{len(chunks)}: cache hit "
                     f"({len(c.text)} chars, audio)")
                pieces.append(hit)
                decorated += 1
                if debug_dir:
                    _emit_debug(
                        debug_dir, c.index,
                        input_text=c.text, output_text=hit,
                        status="cached", reason="",
                    )
                continue

        _log(f"chunk {c.index + 1}/{len(chunks)}: decorating with audio "
             f"({len(c.text)} chars + {len(audio_bytes):,} audio bytes)")
        result = audio_mod.decorate_chunk_with_audio(
            client, c.text,
            audio_bytes=audio_bytes,
            audio_mime=audio_mime,
            model=model,
            strict_tones=strict_tones,
            register=register,
        )
        if result.ok:
            decorated += 1
            if not no_cache:
                cache.put(cdir, cache_key, result.text)
        else:
            passthrough += 1
            _log(f"  passthrough: {result.reason}")
        pieces.append(result.text)
        if debug_dir:
            _emit_debug(
                debug_dir, c.index,
                input_text=c.text, output_text=result.text,
                status=result.status, reason=result.reason,
            )

    return (pieces, decorated, passthrough)


def _process(
    chunks: list[Chunk],
    *,
    no_cache: bool,
    no_llm: bool,
    strict_tones: bool,
    debug_dir: Path | None,
    register: str = "",
) -> tuple[list[str], int, int]:
    """Returns (output_pieces, decorated_count, passthrough_count)."""
    if no_llm:
        for c in chunks:
            _log(f"chunk {c.index + 1}/{len(chunks)}: --no-llm passthrough")
            if debug_dir:
                _emit_debug(
                    debug_dir, c.index,
                    input_text=c.text, output_text=c.text,
                    status="passthrough", reason="--no-llm",
                )
        return ([c.text for c in chunks], 0, len(chunks))

    try:
        api_key = decorate.read_api_key()
    except decorate.MissingAPIKey as e:
        _log(f"warning: {e}; falling back to passthrough")
        return ([c.text for c in chunks], 0, len(chunks))

    client = decorate.make_client(api_key)
    cdir = cache.chunks_dir(_cache_root())
    prompt_hash = decorate.prompt_template_hash()
    model = _model()

    pieces: list[str] = []
    decorated = 0
    passthrough = 0

    for c in chunks:
        cache_key = cache.key_for(
            chunk_text=c.text,
            prev_context=c.prev_context,
            prompt_template_hash=prompt_hash,
            model=model,
            api_version=decorate.ANTHROPIC_API_VERSION,
            register=register,
        )
        if not no_cache:
            hit = cache.get(cdir, cache_key)
            if hit is not None:
                _log(f"chunk {c.index + 1}/{len(chunks)}: cache hit "
                     f"({len(c.text)} chars)")
                pieces.append(hit)
                decorated += 1
                if debug_dir:
                    _emit_debug(
                        debug_dir, c.index,
                        input_text=c.text, output_text=hit,
                        status="cached", reason="",
                    )
                continue

        _log(f"chunk {c.index + 1}/{len(chunks)}: decorating "
             f"({len(c.text)} chars)")
        result = decorate.decorate_chunk(
            client, c.text,
            prev_context=c.prev_context, model=model,
            strict_tones=strict_tones, register=register,
        )
        if result.ok:
            decorated += 1
            if not no_cache:
                cache.put(cdir, cache_key, result.text)
        else:
            passthrough += 1
            _log(f"  passthrough: {result.reason}")
        pieces.append(result.text)
        if debug_dir:
            _emit_debug(
                debug_dir, c.index,
                input_text=c.text, output_text=result.text,
                status=result.status, reason=result.reason,
            )

    return (pieces, decorated, passthrough)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Validate --audio early. Pointing at a missing file is a user
    # error, not a runtime crash — surface as EXIT_INVALID so the
    # listen-tools pipeline can distinguish "you typo'd the path"
    # from "the model went sideways and we passed through".
    if args.audio is not None and not args.audio.is_file():
        _log(f"error: --audio FILE not found: {args.audio}")
        return EXIT_INVALID

    # --no-llm and --audio are mutually exclusive; --audio implies a
    # real LLM call so silently overriding either would surprise the
    # user. Flag explicitly.
    if args.audio is not None and args.no_llm:
        _log("error: --audio and --no-llm are mutually exclusive")
        return EXIT_INVALID

    raw = _read_input(args.input)
    stripped = strip_markdown(raw)
    if not stripped:
        _log("error: no text to decorate")
        return EXIT_EMPTY

    chunks = chunk_markdown(stripped)
    if not chunks:
        _log("error: no chunks after splitting")
        return EXIT_EMPTY

    if args.audio is not None:
        pieces, decorated, passthrough = _process_audio(
            chunks,
            audio_path=args.audio,
            no_cache=args.no_cache,
            strict_tones=args.strict_tones,
            debug_dir=args.debug,
            register=args.register,
        )
    else:
        pieces, decorated, passthrough = _process(
            chunks,
            no_cache=args.no_cache,
            no_llm=args.no_llm,
            strict_tones=args.strict_tones,
            debug_dir=args.debug,
            register=args.register,
        )

    output = "\n\n".join(pieces)
    # Deterministic paragraph-pause enforcement runs on the final
    # assembled text so it sees the FULL article's paragraph boundaries
    # in one pass and stays idempotent across cached/passthrough chunks.
    output = decorate.enforce_paragraph_pauses(output).rstrip() + "\n"
    _write_output(output, args.output)

    _log(f"done. {len(chunks)} chunks ({decorated} decorated, "
         f"{passthrough} passthrough).")
    # Structured stats line for measuring passthrough rate over time
    # (advisor M3). Easy to grep / aggregate later.
    pt_pct = (100 * passthrough / len(chunks)) if chunks else 0
    _log(
        f"stats: chunks={len(chunks)} decorated={decorated} "
        f"passthrough={passthrough} passthrough_pct={pt_pct:.0f} "
        f"strict_tones={int(args.strict_tones)}"
    )

    if decorated == 0 and passthrough > 0:
        return EXIT_FULL_PASSTHROUGH
    if passthrough > 0:
        return EXIT_PARTIAL_PASSTHROUGH
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
