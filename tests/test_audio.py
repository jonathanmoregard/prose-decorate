"""Tests for the Gemini multimodal audio decoration path.

Mirrors tests/test_decorate.py shape; mocks the google-genai client so
no live API call happens in the suite. The provider differences (audio
bytes -> inline Part, system_instruction inside config object, response
shape) are exercised here; everything downstream (_validate, cache,
paragraph-pause enforcement) is shared with the Claude path and stays
covered by test_decorate.py.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from prose_decorate.audio import (
    AUDIO_DEFAULT_MODEL,
    MissingGeminiAPIKey,
    MissingGoogleGenAI,
    audio_hash_for,
    decorate_chunk_with_audio,
    read_gemini_api_key,
)
from prose_decorate.decorate import DecorateStatus


# ---------- API key resolution ----------

def test_read_gemini_api_key_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    monkeypatch.delenv("GEMINI_API_KEY_FILE", raising=False)
    assert read_gemini_api_key() == "g-test"


def test_read_gemini_api_key_file_overrides_env(monkeypatch, tmp_path: Path):
    p = tmp_path / "gkey"
    p.write_text("g-from-file\n")
    monkeypatch.setenv("GEMINI_API_KEY", "g-from-env")
    monkeypatch.setenv("GEMINI_API_KEY_FILE", str(p))
    assert read_gemini_api_key() == "g-from-file"


def test_read_gemini_api_key_missing_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY_FILE", raising=False)
    with pytest.raises(MissingGeminiAPIKey):
        read_gemini_api_key()


# ---------- audio_hash_for ----------

def test_audio_hash_for_is_sha256_hex():
    h = audio_hash_for(b"abc")
    assert len(h) == 64
    # sha256("abc")
    assert h == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_audio_hash_for_differs_for_different_bytes():
    assert audio_hash_for(b"abc") != audio_hash_for(b"xyz")


def test_audio_hash_for_empty_bytes():
    # sha256("") is a well-known constant.
    h = audio_hash_for(b"")
    assert h == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


# ---------- decorate_chunk_with_audio with mocked client ----------

def _fake_gemini_response(text: str):
    """Match google-genai response shape: `response.text` is a str."""
    return SimpleNamespace(text=text)


def test_decorate_with_audio_success_returns_decorated():
    client = MagicMock()
    client.models.generate_content.return_value = _fake_gemini_response(
        "Hello [long pause] world."
    )
    out = decorate_chunk_with_audio(
        client, "Hello world.", audio_bytes=b"FAKE_MP3",
        sleep=lambda _: None,
    )
    assert out.status == DecorateStatus.DECORATED
    assert out.text == "Hello [long pause] world."


def test_decorate_with_audio_passes_audio_bytes_to_client():
    """The audio bytes the caller hands us MUST reach the SDK in the
    contents list; if we silently dropped them we'd be doing text-only
    inference and the whole 'grounded on actual delivery' premise would
    be false."""
    client = MagicMock()
    client.models.generate_content.return_value = _fake_gemini_response(
        "Hello [long pause] world."
    )
    audio = b"\xff\xfb\x90\x00FAKE_MP3_FRAME_DATA"
    decorate_chunk_with_audio(
        client, "Hello world.", audio_bytes=audio, sleep=lambda _: None,
    )

    # Inspect what we passed as `contents` to the SDK.
    call = client.models.generate_content.call_args
    contents = call.kwargs["contents"]
    # contents is a list with at least one Part-shaped object whose
    # raw bytes are the audio. Our wrapper builds Parts via
    # `types.Part.from_bytes(data=audio, mime_type="audio/mp3")` for
    # the audio part. The mock environment doesn't have google-genai
    # imported, so we accept any structure where the audio bytes are
    # discoverable somewhere in the contents list.
    found = _contains_bytes(contents, audio)
    assert found, f"audio bytes not found in contents={contents!r}"


def test_decorate_with_audio_includes_chunk_text_in_contents():
    client = MagicMock()
    client.models.generate_content.return_value = _fake_gemini_response(
        "The chunk text. [long pause] Continues."
    )
    chunk = "The chunk text. Continues."
    decorate_chunk_with_audio(
        client, chunk, audio_bytes=b"AUDIO", sleep=lambda _: None,
    )
    call = client.models.generate_content.call_args
    contents = call.kwargs["contents"]
    assert _contains_substring(contents, chunk), (
        f"chunk text not found in contents={contents!r}"
    )


def test_decorate_with_audio_uses_default_model():
    client = MagicMock()
    client.models.generate_content.return_value = _fake_gemini_response("Hi.")
    decorate_chunk_with_audio(
        client, "Hi.", audio_bytes=b"A", sleep=lambda _: None,
    )
    call = client.models.generate_content.call_args
    assert call.kwargs["model"] == AUDIO_DEFAULT_MODEL
    assert "gemini" in AUDIO_DEFAULT_MODEL.lower()


def test_decorate_with_audio_passes_temperature_zero():
    client = MagicMock()
    client.models.generate_content.return_value = _fake_gemini_response("Hi.")
    decorate_chunk_with_audio(
        client, "Hi.", audio_bytes=b"A", sleep=lambda _: None,
    )
    call = client.models.generate_content.call_args
    config = call.kwargs["config"]
    # config is types.GenerateContentConfig(...) — temperature should be
    # 0 (deterministic) and system_instruction should be set.
    assert _get_attr_or_key(config, "temperature") == 0
    sysi = _get_attr_or_key(config, "system_instruction")
    assert sysi
    assert "Audio" in sysi or "audio" in sysi


def test_decorate_with_audio_system_prompt_mentions_audio_grounding():
    """The audio path must tell the model: base every tag decision on
    what you HEAR, not on inferred-from-prose intuition. If the system
    prompt's audio-grounding clause silently disappears, the path
    degenerates into a more-expensive text-only run."""
    client = MagicMock()
    client.models.generate_content.return_value = _fake_gemini_response("Hi.")
    decorate_chunk_with_audio(
        client, "Hi.", audio_bytes=b"A", sleep=lambda _: None,
    )
    sysi = _get_attr_or_key(
        client.models.generate_content.call_args.kwargs["config"],
        "system_instruction",
    )
    # Some token from the audio-grounding addendum must be present.
    assert "audio" in sysi.lower()
    assert "heard" in sysi.lower() or "hear" in sysi.lower()


def test_decorate_with_audio_register_injects_preamble():
    client = MagicMock()
    client.models.generate_content.return_value = _fake_gemini_response("Hi.")
    decorate_chunk_with_audio(
        client, "Hi.", audio_bytes=b"A",
        register="calm sleepy bedtime", sleep=lambda _: None,
    )
    sysi = _get_attr_or_key(
        client.models.generate_content.call_args.kwargs["config"],
        "system_instruction",
    )
    assert "Register constraint" in sysi
    assert "calm sleepy bedtime" in sysi


def test_decorate_with_audio_passthrough_on_validation_fail():
    """Same word-drift validator as the Claude path. If the model
    rewrites prose, we passthrough."""
    client = MagicMock()
    client.models.generate_content.return_value = _fake_gemini_response(
        "Greetings, earthlings."  # rewrite
    )
    out = decorate_chunk_with_audio(
        client, "Hello world.", audio_bytes=b"A", sleep=lambda _: None,
    )
    assert out.status == DecorateStatus.PASSTHROUGH
    assert out.text == "Hello world."
    assert "drift" in out.reason


def test_decorate_with_audio_passthrough_on_persistent_error():
    client = MagicMock()
    client.models.generate_content.side_effect = RuntimeError("boom")
    out = decorate_chunk_with_audio(
        client, "Hello world.", audio_bytes=b"A",
        sleep=lambda _: None, max_retries=2,
    )
    assert out.status == DecorateStatus.PASSTHROUGH
    assert out.text == "Hello world."


def test_decorate_with_audio_passthrough_on_empty_response():
    client = MagicMock()
    client.models.generate_content.return_value = _fake_gemini_response("")
    out = decorate_chunk_with_audio(
        client, "Hello.", audio_bytes=b"A",
        sleep=lambda _: None, max_retries=2,
    )
    assert out.status == DecorateStatus.PASSTHROUGH


def test_decorate_with_audio_passthrough_on_none_text():
    """SDK may return `response.text is None` if all candidates were
    blocked. Treat as empty -> passthrough."""
    client = MagicMock()
    client.models.generate_content.return_value = _fake_gemini_response(None)
    out = decorate_chunk_with_audio(
        client, "Hello.", audio_bytes=b"A",
        sleep=lambda _: None, max_retries=2,
    )
    assert out.status == DecorateStatus.PASSTHROUGH


# ---------- live SDK shape (skipped if google-genai not installed) ----------

def test_build_contents_uses_real_sdk_types_when_available():
    """When google-genai IS importable, _build_contents must build real
    `types.Part` objects (not the dict fallback). This catches the
    silent-fallback bug where a typo in the import path would make
    every prod call go through the test-only dict shape — Gemini
    would reject those at the wire."""
    pytest.importorskip("google.genai")
    from google.genai import types  # type: ignore[import-not-found]

    from prose_decorate.audio import _build_contents

    contents = _build_contents("Hello.", b"AUDIO", "audio/mp3")
    assert all(isinstance(p, types.Part) for p in contents), (
        f"expected Part instances, got {[type(p).__name__ for p in contents]}"
    )


def test_build_config_uses_real_sdk_types_when_available():
    pytest.importorskip("google.genai")
    from google.genai import types  # type: ignore[import-not-found]

    from prose_decorate.audio import _build_config

    config = _build_config(register="", max_output_tokens=8192)
    assert isinstance(config, types.GenerateContentConfig)


def test_make_client_returns_genai_client_when_available(monkeypatch):
    """End-to-end smoke that the import path is correct. Doesn't make
    any API call — just constructs the client object."""
    pytest.importorskip("google.genai")
    from google import genai  # type: ignore[import-not-found]

    from prose_decorate.audio import make_client

    client = make_client("g-fake-key-for-construction-test")
    assert isinstance(client, genai.Client)


# ---------- helpers ----------

def _get_attr_or_key(obj, name):
    """Look up `name` on `obj` whether it's an attr (Pydantic model /
    dataclass / SimpleNamespace) or a dict key. Tolerates both because
    the SDK constructs config objects, but tests sometimes pass dict
    shapes through MagicMock."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _contains_bytes(structure, needle: bytes) -> bool:
    """Recursively scan a nested list / dict / object structure for
    bytes equal to needle."""
    if isinstance(structure, (bytes, bytearray)):
        return bytes(structure) == needle
    if isinstance(structure, dict):
        return any(_contains_bytes(v, needle) for v in structure.values())
    if isinstance(structure, (list, tuple)):
        return any(_contains_bytes(v, needle) for v in structure)
    # object: walk __dict__ + slots if any
    for attr in ("data", "inline_data", "bytes_value"):
        if hasattr(structure, attr):
            if _contains_bytes(getattr(structure, attr), needle):
                return True
    return False


def _contains_substring(structure, needle: str) -> bool:
    if isinstance(structure, str):
        return needle in structure
    if isinstance(structure, dict):
        return any(_contains_substring(v, needle) for v in structure.values())
    if isinstance(structure, (list, tuple)):
        return any(_contains_substring(v, needle) for v in structure)
    for attr in ("text",):
        if hasattr(structure, attr):
            if _contains_substring(getattr(structure, attr), needle):
                return True
    return False
