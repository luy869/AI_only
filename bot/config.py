"""
Athena Bot — Configuration Module

Loads all settings from environment variables via pydantic-settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
CACHE_DIR = ROOT_DIR / "cache"
LOGS_DIR = ROOT_DIR / "logs"


class BotSettings(BaseSettings):
    """Discord Bot settings."""

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Discord ──────────────────────────────────────────────────────────
    discord_token: str = Field(..., description="Discord Bot Token")
    command_prefix: str = Field("!", description="Legacy prefix (Slash Commands are primary)")
    owner_id: Optional[int] = Field(None, description="Bot owner's Discord user ID")

    # ── LLM (Ollama) ───────────────────────────────────────────────────────
    ollama_base_url: str = Field(
        "http://ollama:11434",
        description="Ollama API base URL",
    )
    ollama_model: str = Field(
        "gemma3:12b-it-qat",
        description="Ollama model name to use for LLM inference",
    )
    llm_max_tokens: int = Field(1024, description="Max tokens to generate per response")
    llm_temperature: float = Field(0.7, description="Sampling temperature")
    llm_top_p: float = Field(0.9, description="Nucleus sampling top-p")
    llm_repeat_penalty: float = Field(1.1, description="Repetition penalty")

    # ── Image Generation ─────────────────────────────────────────────────
    sd_model_id: str = Field(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        description="Hugging Face model ID or local path for Stable Diffusion",
    )
    sd_gpu_device: int = Field(1, description="CUDA device index for SD")
    sd_default_steps: int = Field(25, description="Default inference steps")
    sd_default_guidance: float = Field(7.5, description="Default guidance scale")
    sd_default_width: int = Field(512, description="Default image width")
    sd_default_height: int = Field(512, description="Default image height")
    sd_enable_safety: bool = Field(True, description="Enable NSFW safety checker")
    sd_model_local_path: str = Field(
        str(MODELS_DIR / "stable-diffusion-v1-5"),
        description="Local directory to cache/store SD model",
    )

    # ── Queue / Rate Limit ───────────────────────────────────────────────
    llm_queue_max: int = Field(10, description="Max queued LLM requests")
    image_queue_max: int = Field(5, description="Max queued image requests")
    rate_limit_per_minute: int = Field(10, description="Requests per user per minute")
    rate_limit_burst: int = Field(3, description="Burst allowance")

    # ── Conversation ─────────────────────────────────────────────────────
    max_conversation_turns: int = Field(20, description="Max turns to keep per user")
    max_users_cached: int = Field(200, description="Max users with cached history")

    # ── Cache ─────────────────────────────────────────────────────────────
    image_cache_dir: str = Field(str(CACHE_DIR / "images"), description="Generated image cache dir")
    image_cache_ttl_hours: int = Field(24, description="Image cache TTL in hours")

    # ── Logging ──────────────────────────────────────────────────────────
    log_level: str = Field("INFO", description="Logging level")
    log_file: str = Field(str(LOGS_DIR / "athena.log"), description="Log file path")


# Singleton
settings = BotSettings()  # type: ignore[call-arg]
