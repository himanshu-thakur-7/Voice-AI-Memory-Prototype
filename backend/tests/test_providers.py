"""Provider factory selection — the right STT for the configured key, mock otherwise."""

from __future__ import annotations

from app.config import Settings
from app.providers.factory import make_stt
from app.providers.stt_base import MockSTT


def test_openai_is_default_and_selected_with_key():
    stt = make_stt(Settings(stt_provider="openai", openai_api_key="sk-test"))
    assert stt.name == "openai"  # reuses the OpenAI key — no separate STT account


def test_falls_back_to_mock_without_key():
    stt = make_stt(Settings(stt_provider="openai", openai_api_key=""))
    assert isinstance(stt, MockSTT)


def test_deepgram_selected_with_key():
    stt = make_stt(Settings(stt_provider="deepgram", deepgram_api_key="dg-test"))
    assert stt.name == "deepgram"


def test_default_provider_is_openai():
    # The shipped default reuses the OpenAI key rather than the gated Ringg adapter.
    assert Settings().stt_provider == "openai"
