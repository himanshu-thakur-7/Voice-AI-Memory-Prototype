"""The sentence chunker is what lets TTS start before the LLM finishes."""

from __future__ import annotations

from app.providers.llm_openai import SentenceChunker


def test_emits_sentences_as_boundaries_arrive():
    c = SentenceChunker()
    out: list[str] = []
    out += c.push("Hello there. ")
    out += c.push("How are ")
    out += c.push("you today? ")
    out += c.push("Great")
    assert out == ["Hello there.", "How are you today?"]
    assert c.flush() == "Great"


def test_streaming_token_by_token():
    c = SentenceChunker()
    emitted: list[str] = []
    for tok in "I can help. ":
        emitted += c.push(tok)
    assert emitted == ["I can help."]


def test_handles_multiple_punctuation():
    c = SentenceChunker()
    out = c.push("Really?! Yes. ")
    assert out == ["Really?!", "Yes."]


def test_flush_returns_none_when_empty():
    c = SentenceChunker()
    c.push("Done. ")
    assert c.flush() is None
