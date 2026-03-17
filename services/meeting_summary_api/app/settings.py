from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path


class EnhanceAudioMode(str, Enum):
    OFF = "off"
    ON = "on"
    AUTO = "auto"


# Default bind address inside the container.
DEFAULT_HOST = "0.0.0.0"
# Default internal port exposed by uvicorn.
DEFAULT_PORT = 8080
# Long transcription and summarization requests can legitimately run for many minutes.
DEFAULT_HTTP_TIMEOUT_SECONDS = 1800.0
# Allow dependent model services plenty of time to become ready after container start.
DEFAULT_DEPENDENCY_WAIT_TIMEOUT_SECONDS = 900.0
# Poll upstream dependencies often enough without creating noisy health traffic.
DEFAULT_DEPENDENCY_WAIT_INTERVAL_SECONDS = 5.0
# Reuse the safer ASR chunk size already chosen for GigaAM.
DEFAULT_ASR_CHUNK_SECONDS = 22.0
# Keep the same GigaAM public API bounds in the orchestration layer.
MIN_ASR_CHUNK_SECONDS = 1.0
MAX_ASR_CHUNK_SECONDS = 25.0
# Around 1200 words keeps each LLM chunk comfortably below an 8k context once prompts are added.
DEFAULT_SUMMARY_MAX_WORDS = 1200
# Char cap is a second safety rail for dense transcript fragments.
DEFAULT_SUMMARY_MAX_CHARS = 9000
# Give the model a compact amount of neighbor text for pronouns and dangling references.
DEFAULT_SUMMARY_CONTEXT_WORDS = 250
# Low temperature keeps meeting summaries stable and less verbose.
DEFAULT_LLM_TEMPERATURE = 0.1
# Chunk-level summary budget is enough for a structured fragment summary without rambling.
DEFAULT_LLM_CHUNK_MAX_TOKENS = 900
# Final merge step needs a bit more room to consolidate all partial summaries.
DEFAULT_LLM_FINAL_MAX_TOKENS = 1400


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    work_dir: Path
    gigaam_api_base_url: str
    vllm_api_base_url: str
    vllm_model_name: str
    http_timeout_seconds: float
    dependency_wait_timeout_seconds: float
    dependency_wait_interval_seconds: float
    default_asr_chunk_seconds: float
    min_asr_chunk_seconds: float
    max_asr_chunk_seconds: float
    default_enhance_audio_mode: EnhanceAudioMode
    summary_max_words: int
    summary_max_chars: int
    summary_context_words: int
    llm_temperature: float
    llm_chunk_max_tokens: int
    llm_final_max_tokens: int
    llm_enable_thinking: bool


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    work_dir = Path(os.getenv("APP_WORK_DIR", "/work"))
    (work_dir / "uploads").mkdir(parents=True, exist_ok=True)
    (work_dir / "tmp").mkdir(parents=True, exist_ok=True)

    return Settings(
        host=os.getenv("UVICORN_HOST", DEFAULT_HOST),
        port=int(os.getenv("UVICORN_PORT", str(DEFAULT_PORT))),
        work_dir=work_dir,
        gigaam_api_base_url=os.getenv(
            "GIGAAM_API_BASE_URL",
            "http://gigaam-api:8080",
        ).rstrip("/"),
        vllm_api_base_url=os.getenv(
            "VLLM_API_BASE_URL",
            "http://vllm-qwen:8000/v1",
        ).rstrip("/"),
        vllm_model_name=os.getenv("VLLM_MODEL_NAME", "qwen3-32b-awq"),
        http_timeout_seconds=float(
            os.getenv("HTTP_TIMEOUT_SECONDS", str(DEFAULT_HTTP_TIMEOUT_SECONDS))
        ),
        dependency_wait_timeout_seconds=float(
            os.getenv(
                "DEPENDENCY_WAIT_TIMEOUT_SECONDS",
                str(DEFAULT_DEPENDENCY_WAIT_TIMEOUT_SECONDS),
            )
        ),
        dependency_wait_interval_seconds=float(
            os.getenv(
                "DEPENDENCY_WAIT_INTERVAL_SECONDS",
                str(DEFAULT_DEPENDENCY_WAIT_INTERVAL_SECONDS),
            )
        ),
        default_asr_chunk_seconds=float(
            os.getenv("DEFAULT_ASR_CHUNK_SECONDS", str(DEFAULT_ASR_CHUNK_SECONDS))
        ),
        min_asr_chunk_seconds=float(
            os.getenv("MIN_ASR_CHUNK_SECONDS", str(MIN_ASR_CHUNK_SECONDS))
        ),
        max_asr_chunk_seconds=float(
            os.getenv("MAX_ASR_CHUNK_SECONDS", str(MAX_ASR_CHUNK_SECONDS))
        ),
        default_enhance_audio_mode=EnhanceAudioMode(
            os.getenv(
                "DEFAULT_ENHANCE_AUDIO_MODE",
                EnhanceAudioMode.AUTO.value,
            ).lower()
        ),
        summary_max_words=int(
            os.getenv("SUMMARY_MAX_WORDS", str(DEFAULT_SUMMARY_MAX_WORDS))
        ),
        summary_max_chars=int(
            os.getenv("SUMMARY_MAX_CHARS", str(DEFAULT_SUMMARY_MAX_CHARS))
        ),
        summary_context_words=int(
            os.getenv("SUMMARY_CONTEXT_WORDS", str(DEFAULT_SUMMARY_CONTEXT_WORDS))
        ),
        llm_temperature=float(
            os.getenv("LLM_TEMPERATURE", str(DEFAULT_LLM_TEMPERATURE))
        ),
        llm_chunk_max_tokens=int(
            os.getenv(
                "LLM_CHUNK_MAX_TOKENS",
                str(DEFAULT_LLM_CHUNK_MAX_TOKENS),
            )
        ),
        llm_final_max_tokens=int(
            os.getenv(
                "LLM_FINAL_MAX_TOKENS",
                str(DEFAULT_LLM_FINAL_MAX_TOKENS),
            )
        ),
        llm_enable_thinking=_parse_bool(
            os.getenv("LLM_ENABLE_THINKING"),
            default=False,
        ),
    )
