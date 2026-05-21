from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from prose_decorate.decorate import (
    DecorateStatus,
    MissingAPIKey,
    NonceCollision,
    _build_user_message,
    _validate,
    decorate_chunk,
    prompt_template_hash,
    read_api_key,
)


# ---------- API key resolution ----------

def test_read_api_key_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY_FILE", raising=False)
    assert read_api_key() == "sk-test"


def test_read_api_key_file_overrides_env(monkeypatch, tmp_path: Path):
    p = tmp_path / "key"
    p.write_text("sk-from-file\n")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    monkeypatch.setenv("ANTHROPIC_API_KEY_FILE", str(p))
    assert read_api_key() == "sk-from-file"


def test_read_api_key_missing_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY_FILE", raising=False)
    with pytest.raises(MissingAPIKey):
        read_api_key()


# ---------- nonce + ctx wrapping ----------

def test_build_user_message_no_prev_context():
    msg, nonce = _build_user_message("Hello.", "")
    assert "CHUNK TO DECORATE" in msg
    assert "ctx-prev" not in msg
    assert "Hello." in msg
    assert len(nonce) == 16


def test_build_user_message_with_prev_context_wraps_in_nonce():
    msg, nonce = _build_user_message("Hello.", "Previous para.")
    assert f"<ctx-prev-{nonce}>" in msg
    assert f"</ctx-prev-{nonce}>" in msg
    assert "Previous para." in msg


def test_build_user_message_escapes_lt_in_prev_context():
    msg, _ = _build_user_message("hello.", "Use `<div>` carefully.")
    assert "&lt;div>" in msg
    assert "<div>" not in msg or "<div>" in "</ctx-prev"  # only inside delim


def test_build_user_message_nonce_collision_raises(monkeypatch):
    """Force a known nonce, then feed input containing that nonce string."""
    monkeypatch.setattr(
        "prose_decorate.decorate.secrets.token_hex",
        lambda n: "deadbeef00000000",
    )
    chunk = "Hello <ctx-prev-deadbeef00000000>"
    with pytest.raises(NonceCollision):
        _build_user_message(chunk, "real prev context")


def test_build_user_message_no_collision_when_no_prev_context(monkeypatch):
    """If there's no prev_context, no delimiters are emitted, so collision
    detection doesn't fire even if the chunk literally contains the nonce."""
    monkeypatch.setattr(
        "prose_decorate.decorate.secrets.token_hex",
        lambda n: "deadbeef00000000",
    )
    chunk = "Hello <ctx-prev-deadbeef00000000>"
    msg, _ = _build_user_message(chunk, "")
    assert "Hello" in msg


# ---------- safety guards ----------

def test_validate_accepts_unmodified():
    ok, _ = _validate("Hello world.", "Hello world.")
    assert ok


def test_validate_accepts_tag_insertion():
    ok, _ = _validate("Hello world.", "Hello [short pause] world.")
    assert ok


def test_validate_rejects_prose_drift():
    ok, reason = _validate("Hello world.", "Hello cruel world.")
    assert not ok
    assert "drift" in reason


def test_validate_rejects_excessive_tags():
    # Big enough input to clear the 200-char floor on the tag budget,
    # then output with way too many tags.
    input_text = "Lorem ipsum dolor sit amet. " * 30  # ~840 chars
    output_text = input_text + " " + " ".join([f"[tag{i}]" for i in range(80)])
    ok, reason = _validate(input_text, output_text)
    assert not ok
    assert "tag overhead" in reason


def test_validate_floor_allows_short_input_with_one_tag():
    """Small inputs get a 200-char absolute floor on the tag budget so a
    single tag (~12-20 chars) doesn't blow the 50%-of-input cap."""
    ok, _ = _validate("Hello world.", "Hello [emphasis] world.")
    assert ok


def test_validate_tolerates_whitespace_collapse():
    """LLM frequently joins soft-line-wrapped input. Normalizer collapses."""
    input_text = "Hello  \nworld."
    output_text = "Hello [short pause] world."
    ok, _ = _validate(input_text, output_text)
    assert ok


# ---------- decorate_chunk with mocked client ----------

def _fake_response(text: str):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)]
    )


def test_decorate_chunk_success_returns_decorated():
    client = MagicMock()
    client.messages.create.return_value = _fake_response(
        "Hello [short pause] world."
    )
    out = decorate_chunk(client, "Hello world.", sleep=lambda _: None)
    assert out.status == DecorateStatus.DECORATED
    assert out.text == "Hello [short pause] world."


def test_decorate_chunk_passes_temperature_zero_and_system():
    client = MagicMock()
    client.messages.create.return_value = _fake_response("Hello.")
    decorate_chunk(client, "Hello.", sleep=lambda _: None)
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["temperature"] == 0
    assert kwargs["system"]
    assert kwargs["messages"][0]["role"] == "user"


def test_decorate_chunk_passthrough_on_validation_fail():
    """Model rewrites the prose; decorate must passthrough."""
    client = MagicMock()
    client.messages.create.return_value = _fake_response(
        "Greetings, world."  # rewrite; will fail equality check
    )
    out = decorate_chunk(client, "Hello world.", sleep=lambda _: None)
    assert out.status == DecorateStatus.PASSTHROUGH
    assert out.text == "Hello world."
    assert "drift" in out.reason


def test_decorate_chunk_passthrough_on_persistent_error():
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("non-retriable")
    out = decorate_chunk(
        client, "Hello world.", sleep=lambda _: None, max_retries=3
    )
    assert out.status == DecorateStatus.PASSTHROUGH
    assert out.text == "Hello world."


def test_decorate_chunk_passthrough_on_empty_response():
    client = MagicMock()
    client.messages.create.return_value = _fake_response("")
    out = decorate_chunk(
        client, "Hello.", sleep=lambda _: None, max_retries=2
    )
    assert out.status == DecorateStatus.PASSTHROUGH


def test_prompt_template_hash_is_stable():
    h1 = prompt_template_hash()
    h2 = prompt_template_hash()
    assert h1 == h2
    assert len(h1) == 64
