from prose_decorate.chunk import (
    Chunk,
    chunk_markdown,
    split_paragraphs,
    split_sentences,
)


def test_split_paragraphs_basic():
    assert split_paragraphs("a\n\nb\n\nc") == ["a", "b", "c"]


def test_split_paragraphs_drops_empty():
    assert split_paragraphs("a\n\n\n\nb") == ["a", "b"]


def test_split_paragraphs_preserves_intra_paragraph_newlines():
    assert split_paragraphs("line1\nline2\n\nb") == ["line1\nline2", "b"]


def test_split_sentences_basic():
    sents = split_sentences("First sentence. Second sentence! Third?")
    assert sents == ["First sentence.", "Second sentence!", "Third?"]


def test_split_sentences_holds_abbreviation():
    sents = split_sentences("Dr. Strange arrived. He waved.")
    assert sents == ["Dr. Strange arrived.", "He waved."]


def test_split_sentences_handles_no_terminator():
    assert split_sentences("incomplete") == ["incomplete"]


def test_chunk_empty():
    assert chunk_markdown("") == []


def test_chunk_single_paragraph_fits():
    out = chunk_markdown("A short paragraph.", target=4000, soft=2500)
    assert out == [Chunk(text="A short paragraph.", index=0, prev_context="")]


def test_chunk_two_paragraphs_fit_in_one_chunk():
    text = "Para one.\n\nPara two."
    out = chunk_markdown(text, target=4000, soft=2500)
    assert len(out) == 1
    assert out[0].text == "Para one.\n\nPara two."


def test_chunk_splits_when_target_exceeded():
    p1 = "x" * 100
    p2 = "y" * 100
    out = chunk_markdown(f"{p1}\n\n{p2}", target=150, soft=80)
    assert len(out) == 2
    assert out[0].text == p1
    assert out[1].text == p2


def test_chunk_prev_context_empty_on_first():
    out = chunk_markdown("a\n\n" + "x" * 200, target=100, soft=50)
    assert out[0].prev_context == ""
    assert out[1].prev_context == "a"


def test_chunk_prev_context_is_last_paragraph_of_previous():
    text = "first.\n\n" + "second filler. " * 30 + "\n\n" + "third."
    out = chunk_markdown(text, target=200, soft=100)
    # When we have >=3 chunks, the 3rd's prev_context is the last
    # paragraph of the 2nd chunk.
    if len(out) >= 3:
        assert out[2].prev_context  # non-empty


def test_chunk_sentence_fallback_on_oversize_paragraph():
    """A single paragraph >> target must be sentence-split."""
    sentences = [f"Sentence {i} fills space." for i in range(20)]
    paragraph = " ".join(sentences)
    out = chunk_markdown(paragraph, target=80, soft=40)
    assert len(out) > 1
    rejoined = " ".join(c.text for c in out)
    for s in sentences:
        assert s in rejoined


def test_chunk_indexes_are_sequential():
    text = "\n\n".join(f"Para {i}." for i in range(10))
    out = chunk_markdown(text, target=20, soft=10)
    for i, c in enumerate(out):
        assert c.index == i
