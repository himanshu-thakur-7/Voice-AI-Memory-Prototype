"""Provider construction with graceful mock fallback.

Each factory honors the configured provider when its credentials are present, and
otherwise returns the deterministic mock so the streaming loop always runs. This is what
makes the prototype demoable with an empty .env and unit-testable in CI.
"""

from __future__ import annotations

from app.config import Settings
from app.logging import get_logger
from app.providers.llm_openai import LLM, MockLLM, OpenAILLM
from app.providers.stt_base import MockSTT, STTProvider
from app.providers.tts_elevenlabs import TTS, ElevenLabsTTS, MockTTS

log = get_logger(__name__)


def make_stt(settings: Settings) -> STTProvider:
    choice = settings.stt_provider
    if choice == "openai" and settings.has_openai:
        from app.providers.stt_openai import OpenAITranscribeSTT

        return OpenAITranscribeSTT(settings.openai_api_key, settings.openai_transcribe_model)
    if choice == "ringg" and settings.ringg_api_key:
        from app.providers.stt_ringg import RinggParrotSTT

        return RinggParrotSTT(settings.ringg_api_key, settings.ringg_stt_model)
    if choice == "deepgram" and settings.deepgram_api_key:
        from app.providers.stt_deepgram import DeepgramSTT

        return DeepgramSTT(settings.deepgram_api_key, settings.deepgram_model)
    if choice == "elevenlabs" and settings.elevenlabs_api_key:
        from app.providers.stt_elevenlabs import ElevenLabsScribeSTT

        return ElevenLabsScribeSTT(settings.elevenlabs_api_key)
    log.warning("stt.fallback_mock", requested=choice)
    return MockSTT()


def make_llm(settings: Settings) -> LLM:
    if settings.has_openai:
        return OpenAILLM(settings.openai_api_key, settings.llm_model)
    log.warning("llm.fallback_mock")
    return MockLLM()


def make_tts(settings: Settings) -> TTS:
    if settings.has_elevenlabs:
        return ElevenLabsTTS(
            settings.elevenlabs_api_key, settings.elevenlabs_voice_id, settings.elevenlabs_model
        )
    log.warning("tts.fallback_mock")
    return MockTTS()
