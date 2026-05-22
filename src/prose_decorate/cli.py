"""prose-decorate entrypoint. Markdown -> Fish s2-pro-tagged plain text."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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


def _process(
    chunks: list[Chunk],
    *,
    no_cache: bool,
    no_llm: bool,
    strict_tones: bool,
    debug_dir: Path | None,
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
            strict_tones=strict_tones,
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

    raw = _read_input(args.input)
    stripped = strip_markdown(raw)
    if not stripped:
        _log("error: no text to decorate")
        return EXIT_EMPTY

    chunks = chunk_markdown(stripped)
    if not chunks:
        _log("error: no chunks after splitting")
        return EXIT_EMPTY

    pieces, decorated, passthrough = _process(
        chunks,
        no_cache=args.no_cache,
        no_llm=args.no_llm,
        strict_tones=args.strict_tones,
        debug_dir=args.debug,
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
