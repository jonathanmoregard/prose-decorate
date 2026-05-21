from prose_decorate.markdown_strip import (
    normalize_whitespace,
    strip_markdown,
    strip_tags,
)


def test_strip_fenced_code_block():
    md = "Para one.\n\n```python\nprint('hello')\n```\n\nPara two."
    out = strip_markdown(md)
    assert "(code block omitted)" in out
    assert "print('hello')" not in out
    assert "Para one." in out
    assert "Para two." in out


def test_strip_inline_code_to_token():
    md = "Use `kubectl get pods` to inspect."
    out = strip_markdown(md)
    assert "`" not in out
    assert "(some code)" in out


def test_strip_image_to_alt_text():
    md = "Look: ![A blue cat](https://example.com/cat.png) napping."
    out = strip_markdown(md)
    assert "A blue cat" in out
    assert "example.com" not in out


def test_strip_link_to_text():
    md = "See [the docs](https://example.com/docs) for more."
    out = strip_markdown(md)
    assert "the docs" in out
    assert "example.com" not in out


def test_strip_reference_link():
    md = "See [the docs][1] for more.\n\n[1]: https://example.com"
    out = strip_markdown(md)
    assert "the docs" in out
    assert "example.com" not in out


def test_strip_table_to_token():
    md = (
        "Body before.\n\n"
        "| col1 | col2 |\n"
        "|------|------|\n"
        "| a    | b    |\n"
        "| c    | d    |\n\n"
        "Body after."
    )
    out = strip_markdown(md)
    assert "(table omitted)" in out
    assert "col1" not in out
    assert "a    | b" not in out
    assert "Body before." in out
    assert "Body after." in out


def test_strip_footnote_def_and_ref():
    md = (
        "Sentence with a footnote[^longnote].\n\n"
        "[^longnote]: This is the footnote body."
    )
    out = strip_markdown(md)
    assert "[^longnote]" not in out
    assert "footnote body" not in out
    assert "Sentence with a footnote" in out


def test_strip_heading_hashes_but_keep_text():
    md = "# Title\n\nBody.\n\n## Subhead\n\nMore body."
    out = strip_markdown(md)
    assert "Title" in out
    assert "Subhead" in out
    assert "#" not in out


def test_strip_horizontal_rule():
    md = "Before.\n\n---\n\nAfter."
    out = strip_markdown(md)
    assert "---" not in out
    assert "Before." in out
    assert "After." in out


def test_strip_preserves_emphasis_and_blockquote():
    md = "First **bold** word.\n\n> A quoted aside.\n\nThird *italic* word."
    out = strip_markdown(md)
    assert "**bold**" in out
    assert "*italic*" in out
    assert "> A quoted aside." in out


def test_strip_preserves_em_dash_and_ellipsis():
    md = "He paused — then continued… eventually."
    out = strip_markdown(md)
    assert "—" in out
    assert "…" in out


def test_strip_idempotent_on_plain_text():
    plain = "Plain prose.\n\nSecond paragraph."
    assert strip_markdown(plain) == "Plain prose.\n\nSecond paragraph."


def test_normalize_whitespace_collapses_internal_spaces():
    assert normalize_whitespace("a   b   c") == "a b c"


def test_normalize_whitespace_preserves_paragraph_breaks():
    text = "Para one.\n\nPara two."
    assert normalize_whitespace(text) == text


def test_normalize_whitespace_collapses_triple_newlines():
    assert normalize_whitespace("a\n\n\n\nb") == "a\n\nb"


def test_normalize_whitespace_collapses_soft_newline_to_space():
    """Soft (single) newlines collapse to space; double-newline paragraph
    boundaries are preserved. This is what lets the drift guard tolerate
    Sonnet's habit of joining soft-wrapped input."""
    assert normalize_whitespace("a   \nb   ") == "a b"
    assert normalize_whitespace("a\n\nb") == "a\n\nb"


def test_strip_tags_removes_fish_brackets():
    decorated = "Hello [short pause] world [emphasis] there."
    assert strip_tags(decorated) == "Hello  world  there."


def test_strip_tags_idempotent_on_untagged():
    plain = "Hello world."
    assert strip_tags(plain) == plain


def test_strip_tags_removes_fish_emotion_token():
    decorated = "Use [thoughtfully] this approach."
    assert strip_tags(decorated) == "Use  this approach."


def test_normalize_folds_smart_quotes_to_ascii():
    """Sonnet routinely normalizes curly typography even at temperature=0;
    the drift guard must fold the input the same way."""
    assert normalize_whitespace("don\u2019t") == "don't"
    assert normalize_whitespace("\u201chello\u201d") == "\"hello\""
    assert normalize_whitespace("dash\u2014here") == "dash-here"
    assert normalize_whitespace("etc\u2026") == "etc..."


def test_normalize_strips_markdown_emphasis_at_word_boundary():
    """The system prompt tells the LLM to drop markdown emphasis syntax
    after deciding whether to tag it. Normalizer must drop the same
    markers from the input side for a fair comparison."""
    assert normalize_whitespace("**bold** word") == "bold word"
    assert normalize_whitespace("*Summary:* text") == "Summary: text"
    assert normalize_whitespace("__strong__ run") == "strong run"


def test_normalize_strips_leading_whitespace_on_quote_lines():
    """When the LLM emits a quoted line with a stray leading space the
    drift guard should still match the un-spaced input."""
    inp = "Bruce says:\n\nThe Method is..."
    out = "Bruce says:\n\n The Method is..."
    assert normalize_whitespace(inp) == normalize_whitespace(out)


def test_strip_bare_http_url():
    md = "See https://example.com/foo/bar for details."
    out = strip_markdown(md)
    assert "https" not in out
    assert "example.com" not in out
    assert "(a link)" in out
    assert "See" in out and "for details" in out


def test_strip_bare_url_with_trailing_period():
    md = "Visit https://example.com/page."
    out = strip_markdown(md)
    assert "https" not in out
    # Sentence-final period preserved (URL regex doesn't consume it)
    assert out.rstrip().endswith(".")


def test_strip_preserves_inline_url_in_markdown_link():
    """If the URL was already inside `[text](url)`, the link rewrite
    drops it before the bare-URL pass sees it — text only remains."""
    md = "See [the docs](https://example.com/docs) here."
    out = strip_markdown(md)
    assert "https" not in out
    assert "example.com" not in out
    assert "the docs" in out
    # `(a link)` placeholder should NOT appear — the markdown-link
    # rewrite already extracted the text, so the bare-URL pass sees
    # nothing left to substitute.
    assert "(a link)" not in out
