"""Typed application settings.

Everything is environment-driven (12-factor). The prototype runs with an empty
environment: any provider whose key is missing falls back to a deterministic MOCK,
so the whole streaming loop is exercisable offline.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

EngineName = Literal["cascade", "realtime"]
STTName = Literal["openai", "deepgram", "elevenlabs", "ringg"]
AffectExtractor = Literal["auto", "rich", "heuristic"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── engine / provider selection ──────────────────────────────────────────
    engine: EngineName = "cascade"
    # Default STT reuses the OpenAI key (no separate STT vendor/account). Swap to
    # deepgram / elevenlabs / ringg via STT_PROVIDER.
    stt_provider: STTName = "openai"

    # ── server wiring ────────────────────────────────────────────────────────
    backend_listen_addr: str = "0.0.0.0:8000"
    grpc_listen_addr: str = "0.0.0.0:50051"
    log_level: str = "info"

    # ── OpenAI ───────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    llm_model: str = "gpt-4o"
    openai_realtime_model: str = "gpt-realtime"
    openai_transcribe_model: str = "gpt-4o-transcribe"  # realtime STT model
    openai_lexical_model: str = "gpt-4o-mini"           # post-call contradiction/tone pass

    # ── post-call affective extraction ───────────────────────────────────────
    # auto: rich SenseVoice+openSMILE engine if its deps are installed, else heuristic.
    # rich: force the rich engine (warns + falls back if deps missing).
    # heuristic: always use the dependency-free heuristic.
    affect_extractor: AffectExtractor = "auto"

    # ── Ringg Parrot STT (default-of-record) ─────────────────────────────────
    ringg_api_key: str = ""
    ringg_stt_model: str = "parrot-stt-v1"

    # ── Deepgram (runnable fallback) ─────────────────────────────────────────
    deepgram_api_key: str = ""
    deepgram_model: str = "nova-3"

    # ── ElevenLabs ───────────────────────────────────────────────────────────
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = ""
    elevenlabs_model: str = "eleven_flash_v2_5"

    # ── memory stores ────────────────────────────────────────────────────────
    database_url: str = "postgresql://voiceai:voiceai@localhost:5432/voiceai"
    falkordb_host: str = "localhost"
    falkordb_port: int = 6379
    redis_url: str = "redis://localhost:6379/0"
    mem0_graph_enabled: bool = False

    # ── prompts ──────────────────────────────────────────────────────────────
    base_system_prompt: str = (
        "You are a warm, concise voice assistant on a phone call. "
        "Keep replies to one or two sentences. Never use markdown or emoji."
    )

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def has_elevenlabs(self) -> bool:
        return bool(self.elevenlabs_api_key and self.elevenlabs_voice_id)


@lru_cache
def get_settings() -> Settings:
    return Settings()
