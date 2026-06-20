"""LLM leg: GPT-4o token streaming + early sentence chunking.

The latency game (docs/latency-budget.md): never wait for the full completion. Stream
tokens, cut at the FIRST sentence boundary, and hand that sentence to TTS immediately so
the caller hears audio while the model is still generating the rest.

``LLM_MODEL`` is configurable; GPT-4o is honored per spec but is being superseded —
``gpt-4o-mini`` / newer models have materially lower time-to-first-token.
"""

from __future__ import annotations

import abc
import asyncio
from collections.abc import AsyncIterator

from app.metrics import TurnTimer

# A short, voice-shaped reply keeps both latency and token cost down.
MAX_REPLY_TOKENS = 160


class SentenceChunker:
    """Accumulates streamed tokens and emits complete sentences as soon as they form.

    Pure and synchronous → unit-testable in isolation (tests/test_chunker.py).
    """

    _BOUNDARIES = ".!?"

    def __init__(self, min_chars: int = 2) -> None:
        self._buf = ""
        self._min = min_chars

    def push(self, text: str) -> list[str]:
        self._buf += text
        out: list[str] = []
        while (idx := self._boundary(self._buf)) != -1:
            sentence = self._buf[: idx + 1].strip()
            self._buf = self._buf[idx + 1 :]
            if sentence:
                out.append(sentence)
        return out

    def flush(self) -> str | None:
        s = self._buf.strip()
        self._buf = ""
        return s or None

    def _boundary(self, s: str) -> int:
        for i, ch in enumerate(s):
            if ch in self._BOUNDARIES and i + 1 < len(s) and s[i + 1] in " \n\t":
                if i + 1 >= self._min:
                    return i
        return -1


class LLM(abc.ABC):
    name = "base"

    @abc.abstractmethod
    def stream_sentences(
        self, messages: list[dict], timer: TurnTimer
    ) -> AsyncIterator[str]:
        """Yield reply sentences as they become available."""
        raise NotImplementedError


class OpenAILLM(LLM):
    name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        self._api_key = api_key
        self._model = model

    async def stream_sentences(
        self, messages: list[dict], timer: TurnTimer
    ) -> AsyncIterator[str]:
        from openai import AsyncOpenAI  # lazy

        client = AsyncOpenAI(api_key=self._api_key)
        chunker = SentenceChunker()
        stream = await client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
            max_tokens=MAX_REPLY_TOKENS,
            temperature=0.6,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if not delta:
                continue
            timer.mark_llm_first_token()
            for sentence in chunker.push(delta):
                yield sentence
        if tail := chunker.flush():
            yield tail


class MockLLM(LLM):
    """Deterministic templated reply so the loop runs without an OpenAI key."""

    name = "mock"

    async def stream_sentences(
        self, messages: list[dict], timer: TurnTimer
    ) -> AsyncIterator[str]:
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        reply = (
            f"Got it. You said: {user}. "
            "Let me help you with that right away."
        )
        chunker = SentenceChunker()
        # Simulate token streaming so first-token timing is meaningful.
        for word in reply.split(" "):
            await asyncio.sleep(0.005)
            timer.mark_llm_first_token()
            for sentence in chunker.push(word + " "):
                yield sentence
        if tail := chunker.flush():
            yield tail
