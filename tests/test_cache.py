from pathlib import Path

from prose_decorate import cache


def test_key_deterministic():
    args = dict(
        chunk_text="hello",
        prev_context="",
        prompt_template_hash="abc",
        model="claude-sonnet-4-6",
        api_version="2025-09-01",
    )
    assert cache.key_for(**args) == cache.key_for(**args)


def test_key_changes_with_chunk_text():
    base = dict(
        chunk_text="hello", prev_context="",
        prompt_template_hash="h", model="m", api_version="v",
    )
    assert cache.key_for(**base) != cache.key_for(**{**base, "chunk_text": "HELLO"})


def test_key_changes_with_prev_context():
    base = dict(
        chunk_text="hi", prev_context="",
        prompt_template_hash="h", model="m", api_version="v",
    )
    assert cache.key_for(**base) != cache.key_for(**{**base, "prev_context": "x"})


def test_key_changes_with_prompt_hash():
    base = dict(
        chunk_text="hi", prev_context="x",
        prompt_template_hash="h1", model="m", api_version="v",
    )
    assert cache.key_for(**base) != cache.key_for(**{**base, "prompt_template_hash": "h2"})


def test_key_changes_with_model():
    base = dict(
        chunk_text="hi", prev_context="x",
        prompt_template_hash="h", model="claude-sonnet-4-6", api_version="v",
    )
    assert cache.key_for(**base) != cache.key_for(**{**base, "model": "claude-opus-4-7"})


def test_key_changes_with_api_version():
    base = dict(
        chunk_text="hi", prev_context="x",
        prompt_template_hash="h", model="m", api_version="2025-09-01",
    )
    assert cache.key_for(**base) != cache.key_for(**{**base, "api_version": "2026-01-01"})


def test_roundtrip(tmp_path: Path):
    cdir = cache.chunks_dir(tmp_path)
    key = cache.key_for(
        chunk_text="hi", prev_context="",
        prompt_template_hash="h", model="m", api_version="v",
    )
    assert cache.get(cdir, key) is None
    cache.put(cdir, key, "decorated text [emphasis] here")
    assert cache.get(cdir, key) == "decorated text [emphasis] here"


def test_put_atomic_no_tmp_left(tmp_path: Path):
    cdir = cache.chunks_dir(tmp_path)
    cache.put(cdir, "k", "data")
    assert list(cdir.glob("*.tmp")) == []


def test_key_changes_with_register():
    base = dict(
        chunk_text="hi", prev_context="",
        prompt_template_hash="h", model="m", api_version="v",
    )
    base_key = cache.key_for(**base)
    sleepy_key = cache.key_for(**base, register="calm sleepy bedtime")
    firm_key = cache.key_for(**base, register="firm matter-of-fact")
    assert base_key != sleepy_key
    assert sleepy_key != firm_key


def test_key_register_empty_matches_unset():
    base = dict(
        chunk_text="hi", prev_context="",
        prompt_template_hash="h", model="m", api_version="v",
    )
    assert cache.key_for(**base) == cache.key_for(**base, register="")


def test_key_changes_with_audio_hash():
    """Audio runs must invalidate independently from text-only runs.
    Same chunk + register + model but different audio bytes -> different key."""
    base = dict(
        chunk_text="hi", prev_context="",
        prompt_template_hash="h", model="m", api_version="v",
    )
    base_key = cache.key_for(**base)
    audio_a = cache.key_for(**base, audio_hash="a" * 64)
    audio_b = cache.key_for(**base, audio_hash="b" * 64)
    assert base_key != audio_a
    assert audio_a != audio_b


def test_key_audio_hash_empty_matches_unset():
    base = dict(
        chunk_text="hi", prev_context="",
        prompt_template_hash="h", model="m", api_version="v",
    )
    assert cache.key_for(**base) == cache.key_for(**base, audio_hash="")
