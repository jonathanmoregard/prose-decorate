"""Content-hashed disk cache for decorated chunks.

Mirrors tts-tool/src/tts_tool/cache.py shape: sha256 key, atomic write,
no eviction. Difference is the value is text not bytes, and the key
includes the prompt-template hash + model + API version so prompt
iteration and model swaps invalidate cleanly.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from platformdirs import user_cache_dir


def default_cache_dir() -> Path:
    return Path(user_cache_dir("prose-decorate"))


def chunks_dir(override: str | os.PathLike[str] | None = None) -> Path:
    base = Path(override) if override else default_cache_dir()
    out = base / "chunks"
    out.mkdir(parents=True, exist_ok=True)
    return out


def key_for(
    *,
    chunk_text: str,
    prev_context: str,
    prompt_template_hash: str,
    model: str,
    api_version: str,
    register: str = "",
) -> str:
    """`register` is the --register hint string; included in the key so
    swapping register invalidates without manual --no-cache."""
    payload = "|".join([
        chunk_text,
        prev_context,
        prompt_template_hash,
        model,
        api_version,
        register,
    ]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def path_for(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.txt"


def get(cache_dir: Path, key: str) -> str | None:
    p = path_for(cache_dir, key)
    return p.read_text(encoding="utf-8") if p.exists() else None


def put(cache_dir: Path, key: str, decorated: str) -> None:
    p = path_for(cache_dir, key)
    tmp = p.with_suffix(".txt.tmp")
    tmp.write_text(decorated, encoding="utf-8")
    os.replace(tmp, p)
