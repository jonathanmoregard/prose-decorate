from prose_decorate.markdown_strip import (
    normalize_whitespace,
    strip_markdown,
    strip_tags,
)


def test_strip_fenced_code_block():
    md = "Para one.\n\n```python\nprint('hello')\n```\n\nPara two."
    out = strip_markdown(md)
    assert "[code omitted]" in out
    assert "print('hello')" not in out
    assert "Para one." in out
    assert "Para two." in out


def test_strip_inline_code_to_token():
    md = "Use `kubectl get pods` to inspect."
    out = strip_markdown(md)
    assert "`" not in out
    assert "[code]" in out


def test_strip_image_to_alt_text():
    md = "Look: ![A blue cat](https://example.com/cat.png) napping."
    out = strip_markdown(md)
    assert "A blue cat" in out
    assert "example.com" not in out
    assert "!" not in out or "[code]" not in out  # no leftover image syntax


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
    assert "[table omitted]" in out
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


def test_strip_tags_removes_code_token():
    decorated = "Use [code] to inspect."
    assert strip_tags(decorated) == "Use  to inspect."
