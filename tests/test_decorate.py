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
    ok, _ = _validate("Hello world.", "Hello [long pause] world.")
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
    output_text = "Hello [long pause] world."
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
        "Hello [long pause] world."
    )
    out = decorate_chunk(client, "Hello world.", sleep=lambda _: None)
    assert out.status == DecorateStatus.DECORATED
    assert out.text == "Hello [long pause] world."


def test_decorate_chunk_passes_temperature_zero_and_system():
    client = MagicMock()
    client.messages.create.return_value = _fake_response("Hello.")
    decorate_chunk(client, "Hello.", sleep=lambda _: None)
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["temperature"] == 0
    assert kwargs["system"]
    assert kwargs["messages"][0]["role"] == "user"


def test_decorate_chunk_register_injects_preamble_into_system():
    """`register` shows up as a Register-constraint preamble in the system prompt."""
    client = MagicMock()
    client.messages.create.return_value = _fake_response("Hello.")
    decorate_chunk(
        client, "Hello.",
        register="calm, meandering, sleepy bedtime narration",
        sleep=lambda _: None,
    )
    sys_prompt = client.messages.create.call_args.kwargs["system"]
    assert "Register constraint" in sys_prompt
    assert "calm, meandering, sleepy bedtime narration" in sys_prompt
    # Base prompt still present
    assert "## Tag reference" in sys_prompt


def test_decorate_chunk_empty_register_leaves_base_prompt():
    client = MagicMock()
    client.messages.create.return_value = _fake_response("Hello.")
    decorate_chunk(client, "Hello.", register="", sleep=lambda _: None)
    sys_prompt = client.messages.create.call_args.kwargs["system"]
    assert "Register constraint" not in sys_prompt
    assert "## Tag reference" in sys_prompt


def test_decorate_chunk_whitespace_only_register_treated_as_empty():
    client = MagicMock()
    client.messages.create.return_value = _fake_response("Hello.")
    decorate_chunk(client, "Hello.", register="   \n  ", sleep=lambda _: None)
    sys_prompt = client.messages.create.call_args.kwargs["system"]
    assert "Register constraint" not in sys_prompt


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


def test_validate_tolerates_dedup_of_doubled_line():
    """When the input had a duplicated title (substack-url-tool v0.1 bug)
    and the LLM helpfully drops the duplicate, the word-level guard
    sees the same words on both sides minus the title's words once.
    That IS a drift (words removed), and we want to catch it — but
    the symmetric case where the LLM ADDS words is what matters most.
    Just sanity-check that pure-passthrough (no edits) accepts."""
    text = "A title here.\n\nA title here.\n\nBody."
    ok, _ = _validate(text, text)
    assert ok


def test_validate_tolerates_run_together_sentence_split():
    """Claude often inserts a space between run-together sentences
    (`true.Source:` -> `true. Source:`). Word stream is identical;
    only whitespace differs. Word-level guard must accept."""
    ok, _ = _validate(
        "One belief: too good to be true.Source: page 31.",
        "One belief: too good to be true. Source: page 31.",
    )
    assert ok


def test_validate_tolerates_case_change_at_sentence_boundary():
    """At sentence wrap the LLM may re-cap. Word-stream casefolds."""
    ok, _ = _validate("hello world.", "Hello world.")
    assert ok


def test_validate_catches_real_word_substitution():
    ok, reason = _validate("Hello world.", "Hello cruel world.")
    assert not ok
    assert "drift" in reason


# ---------- tone-pairing guard (--strict-tones) ----------

def test_tone_pairing_accepts_opener_with_close():
    ok, _ = _validate(
        "Body.\n\n> Quoted line.",
        "Body. [reading aloud] Quoted line. [back to narration]",
        strict_tones=True,
    )
    assert ok


def test_tone_pairing_rejects_unclosed_opener():
    ok, reason = _validate(
        "Body.\n\nFinal paragraph.",
        "Body. [reading aloud] Final paragraph.",
        strict_tones=True,
    )
    assert not ok
    assert "unclosed" in reason


def test_tone_pairing_accepts_stacked_opener_as_auto_close():
    """S2-Pro semantics: a new opener overrides the previous (advisor H2)."""
    ok, _ = _validate(
        "Body.\n\nQuote a.\n\n> Quote b.",
        "Body. [reading aloud] Quote a. [thoughtfully] Quote b. [back to narration]",
        strict_tones=True,
    )
    assert ok


def test_tone_pairing_rejects_stray_closer():
    ok, reason = _validate(
        "Hello world.",
        "Hello [back to narration] world.",
        strict_tones=True,
    )
    assert not ok
    assert "stray" in reason or "no opener" in reason


def test_tone_pairing_transient_doesnt_count_as_opener():
    """Per prompt's `Tonal voice tags vs transient cues` section,
    only the closed _TONE_OPENERS set is treated as persistent. Free-
    form adverbial cues like `[after a moment]` are transient."""
    ok, _ = _validate(
        "First sentence.\n\nSecond sentence.",
        "First sentence. [after a moment] Second sentence.",
        strict_tones=True,
    )
    assert ok


def test_tone_pairing_closer_substring_match():
    """Advisor M2: accept minor closer variants via substring."""
    ok, _ = _validate(
        "Body.\n\n> Quote.",
        "Body. [reading aloud] Quote. [narrator returns]",
        strict_tones=True,
    )
    assert ok


def test_tone_pairing_allows_unclosed_if_input_ends_in_blockquote():
    """Advisor M1: when the chunk's input itself ends mid-quote, an
    unclosed tone at chunk-end is legitimate — the NEXT chunk continues
    the quote and emits the closer when the quote actually ends."""
    ok, _ = _validate(
        "Body.\n\n> Quoted line that spans into next chunk.",
        "Body. [reading aloud] Quoted line that spans into next chunk.",
        strict_tones=True,
    )
    assert ok


def test_tone_pairing_strict_mode_off_by_default():
    """Default behavior: tone-pairing not enforced; word-drift still is."""
    ok, _ = _validate(
        "Body.\n\nFinal.",
        "Body. [reading aloud] Final.",
        # strict_tones defaults to False
    )
    assert ok


def test_tone_pairing_case_insensitive_and_whitespace_tolerant():
    """Advisor L3: `.strip().casefold()` before matching."""
    ok, _ = _validate(
        "Body.\n\n> Quote.",
        "Body. [ Reading Aloud ] Quote. [ Back to Narration ]",
        strict_tones=True,
    )
    assert ok


def test_tone_pairing_rejects_empty_body_tag():
    """Round-2 M2: `[ ]` would render as literal brackets in audio."""
    ok, reason = _validate("Hello.", "Hello [ ] world." if False else "Hello [ ] there.", strict_tones=True)
    # Word stream of "Hello [ ] there." vs input "Hello." differs in
    # word count. Test goes through tone-pairing first (validate order)
    # and rejects on empty body. Use input that has matching word stream:
    ok, reason = _validate("Hello.", "Hello[ ].", strict_tones=True)
    assert not ok
    assert "empty" in reason


def test_tone_pairing_descriptive_narrator_tag_not_closer():
    """Round-2 M1: tighter closer phrases. Free-form `[narrator pauses]`
    isn't a closer; it's descriptive, treated as transient."""
    ok, reason = _validate(
        "Body.\n\nFinal.",
        "Body. [reading aloud] Final. [narrator pauses]",
        strict_tones=True,
    )
    # `[narrator pauses]` doesn't match the closer phrase list, so
    # `[reading aloud]` remains unclosed -> reject.
    assert not ok
    assert "unclosed" in reason


def test_tone_pairing_chunk_boundary_synthetic_regression():
    """Round-2 L2: stress the chunk-boundary path explicitly."""
    chunk = "Body.\n\n> The quote begins and continues into the next chunk."
    # Input ends in a blockquote — unclosed opener allowed.
    ok, _ = _validate(
        chunk,
        "Body. [reading aloud] The quote begins and continues into the next chunk.",
        strict_tones=True,
    )
    assert ok


# ---------- paragraph-pause enforcement ----------

def test_paragraph_pauses_added_to_each_boundary_except_final():
    """Advisor round-1 H2: final paragraph is intentionally skipped
    (no beat needed at EOF)."""
    from prose_decorate.decorate import enforce_paragraph_pauses
    text = "Para one.\n\nPara two.\n\nPara three."
    out = enforce_paragraph_pauses(text)
    # Paragraphs 1 and 2 get pauses; paragraph 3 (final) does not.
    assert out.count("[long pause]") == 2
    assert out.rstrip().endswith("Para three.")


def test_paragraph_pauses_skip_already_paused_tail():
    from prose_decorate.decorate import enforce_paragraph_pauses
    # `[short pause]` on para 1's tail keeps the enforcer from stacking
    # a `[long pause]` on top of it. Use the SHORT variant in input so
    # the count test distinguishes pre-existing vs newly-inserted.
    text = "Para one [short pause]\n\nPara two.\n\nPara three."
    out = enforce_paragraph_pauses(text)
    # Para one already ends in a pause-shaped tag (skip), Para two gets
    # one inserted, Para three is final (skip) -> exactly 1 [long pause]
    # inserted; the original [short pause] preserved.
    assert out.count("[long pause]") == 1
    assert "[short pause]" in out


def test_paragraph_pauses_appended_after_tone_closer():
    """Tone closers do NOT produce silence, so paragraph-pause
    enforcement MUST still fire after them. Earlier 'skip-on-closer'
    behavior caused real-article bleed (2026-05-22): paragraph ending
    in `[back to narration]` ran directly into the next without a beat."""
    from prose_decorate.decorate import enforce_paragraph_pauses
    text = "Quote ended. [back to narration]\n\nMid para.\n\nFinal."
    out = enforce_paragraph_pauses(text)
    # Para 1 (ends in closer) gets a pause, Para 2 gets a pause,
    # Para 3 final (skip) -> 2 pauses total.
    assert out.count("[long pause]") == 2
    assert "[back to narration]" in out
    assert out.rstrip().endswith("Final.")


def test_paragraph_pauses_idempotent():
    from prose_decorate.decorate import enforce_paragraph_pauses
    text = "Para one.\n\nPara two.\n\nPara three."
    once = enforce_paragraph_pauses(text)
    twice = enforce_paragraph_pauses(once)
    assert once == twice


def test_paragraph_pauses_recognizes_beat_and_breath():
    from prose_decorate.decorate import enforce_paragraph_pauses
    text = "Para one [beat]\n\nPara two [breath]\n\nPara three.\n\nFinal."
    out = enforce_paragraph_pauses(text)
    # 1 & 2 already paused; 3 gets a pause; final skipped.
    assert out.count("[long pause]") == 1


def test_paragraph_pauses_handles_blank_paragraphs():
    from prose_decorate.decorate import enforce_paragraph_pauses
    text = "Para one.\n\n\n\nPara two.\n\nFinal."  # extra blank line in middle
    out = enforce_paragraph_pauses(text)
    assert "Para one." in out
    assert "Para two." in out
    assert "[long pause]" in out


def test_paragraph_pauses_single_paragraph_no_pause():
    """If there's only one paragraph, the final-paragraph rule means
    no pause anywhere."""
    from prose_decorate.decorate import enforce_paragraph_pauses
    text = "Only one paragraph here."
    out = enforce_paragraph_pauses(text)
    assert "[long pause]" not in out
