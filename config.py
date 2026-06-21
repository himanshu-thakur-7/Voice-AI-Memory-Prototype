"""Typed settings loaded from environment (.env).

Pydantic-settings reads ``.env`` if present; every field has a sensible default so the
worker still boots when keys are missing (live components downshift to mock paths).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

AffectExtractor = Literal["auto", "rich", "librosa"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── LiveKit ──────────────────────────────────────────────────────────────
    livekit_url: str = "ws://localhost:7880"
    livekit_api_key: str = "devkey"
    livekit_api_secret: str = "devsecret123456789012345678901234"

    # ── OpenAI ───────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    llm_model: str = "gpt-4o"
    lexical_model: str = "gpt-4o-mini"

    # ── ElevenLabs ───────────────────────────────────────────────────────────
    elevenlabs_api_key: str = ""
    # Deepgram Aura TTS voice models. Switched here from ElevenLabs after both flash
    # and turbo models reproducibly returned "no audio frames were pushed" on the
    # cold-start neutral prosody profile (silent calls, three retries, give up).
    # The two slots differentiate Dynamic Prosody: cold-start uses a brisker voice,
    # frustrated/angry callers route to a warmer/calmer one. The field names start
    # with elevenlabs_ for historic compatibility with the existing .env layout — they
    # carry Aura model strings now (e.g. "aura-2-andromeda-en"), not ElevenLabs IDs.
    elevenlabs_default_voice_id: str = "aura-2-andromeda-en"   # brisk-warm female
    elevenlabs_calm_voice_id:    str = "aura-2-asteria-en"     # calmer, more measured female

    # ── Deepgram ─────────────────────────────────────────────────────────────
    deepgram_api_key: str = ""
    deepgram_model: str = "nova-3"

    # ── Neo4j ────────────────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "cognitive-dev-password"

    # ── Acoustic engine ──────────────────────────────────────────────────────
    affect_extractor: AffectExtractor = "auto"

    # ── Adaptive verbosity ───────────────────────────────────────────────────
    impatience_threshold: int = 2

    # ── Base prompt that pre-call decorates with prosody/empathy/verbosity. ──
    base_system_prompt: str = (
        "You are a warm, concise voice assistant on a phone call. "
        "Keep replies to one or two sentences. Never use markdown or emoji."
    )

    # ── Web demo (web/server.py) ────────────────────────────────────────────
    web_host: str = "0.0.0.0"
    web_port: int = 8000

    # ── Convenience flags ────────────────────────────────────────────────────
    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def has_elevenlabs(self) -> bool:
        return bool(self.elevenlabs_api_key)

    @property
    def has_deepgram(self) -> bool:
        return bool(self.deepgram_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
