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


# Default API bind address inside the container.
DEFAULT_HOST = "0.0.0.0"
# Default internal container port for uvicorn.
DEFAULT_PORT = 8080
# GigaAM short-form transcription is limited to 25 seconds; 22 seconds leaves headroom.
DEFAULT_CHUNK_SECONDS = 22.0
# Do not allow zero-length or near-zero chunks from the public API.
MIN_CHUNK_SECONDS = 1.0
# Hard upper bound supported by `model.transcribe()`.
MAX_CHUNK_SECONDS = 25.0
# Speech quieter than this is usually worth normalizing or enhancing.
DEFAULT_QUIET_SPEECH_DBFS = -30.0
# Minimum desirable gap between speech and pauses/background noise.
DEFAULT_SNR_THRESHOLD_DB = 12.0
# Flag clipping once more than 0.5% of samples are near full scale.
DEFAULT_CLIPPING_RATIO_THRESHOLD = 0.005
# Warn when actual speech occupies less than 20% of the file.
DEFAULT_LOW_SPEECH_RATIO_THRESHOLD = 0.20
# Auto-enhancement turns on once the heuristic score reaches this threshold.
DEFAULT_ENABLE_THRESHOLD = 1.0


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    device: str
    gigaam_model: str
    silero_vad_repo_dir: Path
    work_dir: Path
    model_cache_dir: Path
    torch_cache_dir: Path
    default_chunk_seconds: float
    min_chunk_seconds: float
    max_chunk_seconds: float
    default_enhance_audio_mode: EnhanceAudioMode
    auto_quality_quiet_speech_dbfs: float
    auto_quality_snr_threshold_db: float
    auto_quality_clipping_ratio_threshold: float
    auto_quality_low_speech_ratio_threshold: float
    auto_quality_enable_threshold: float


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    work_dir = Path(os.getenv("APP_WORK_DIR", "/work"))
    model_cache_dir = Path(os.getenv("GIGAAM_CACHE_DIR", "/models/gigaam"))
    torch_cache_dir = Path(os.getenv("TORCH_HOME", "/models/torch"))

    for path in (work_dir, model_cache_dir, torch_cache_dir, work_dir / "tmp"):
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TORCH_HOME", str(torch_cache_dir))
    default_enhance_audio_mode = EnhanceAudioMode(
        os.getenv("DEFAULT_ENHANCE_AUDIO_MODE", EnhanceAudioMode.AUTO.value).lower()
    )

    return Settings(
        host=os.getenv("UVICORN_HOST", DEFAULT_HOST),
        port=int(os.getenv("UVICORN_PORT", str(DEFAULT_PORT))),
        device=os.getenv("APP_DEVICE", "cpu"),
        gigaam_model=os.getenv("GIGAAM_MODEL", "v3_e2e_rnnt"),
        silero_vad_repo_dir=Path(os.getenv("SILERO_VAD_REPO_DIR", "/opt/silero-vad")),
        work_dir=work_dir,
        model_cache_dir=model_cache_dir,
        torch_cache_dir=torch_cache_dir,
        default_chunk_seconds=float(
            os.getenv("DEFAULT_CHUNK_SECONDS", str(DEFAULT_CHUNK_SECONDS))
        ),
        min_chunk_seconds=float(
            os.getenv("MIN_CHUNK_SECONDS", str(MIN_CHUNK_SECONDS))
        ),
        max_chunk_seconds=float(
            os.getenv("MAX_CHUNK_SECONDS", str(MAX_CHUNK_SECONDS))
        ),
        default_enhance_audio_mode=default_enhance_audio_mode,
        auto_quality_quiet_speech_dbfs=float(
            os.getenv(
                "AUTO_QUALITY_QUIET_SPEECH_DBFS",
                str(DEFAULT_QUIET_SPEECH_DBFS),
            )
        ),
        auto_quality_snr_threshold_db=float(
            os.getenv("AUTO_QUALITY_SNR_THRESHOLD_DB", str(DEFAULT_SNR_THRESHOLD_DB))
        ),
        auto_quality_clipping_ratio_threshold=float(
            os.getenv(
                "AUTO_QUALITY_CLIPPING_RATIO_THRESHOLD",
                str(DEFAULT_CLIPPING_RATIO_THRESHOLD),
            )
        ),
        auto_quality_low_speech_ratio_threshold=float(
            os.getenv(
                "AUTO_QUALITY_LOW_SPEECH_RATIO_THRESHOLD",
                str(DEFAULT_LOW_SPEECH_RATIO_THRESHOLD),
            )
        ),
        auto_quality_enable_threshold=float(
            os.getenv("AUTO_QUALITY_ENABLE_THRESHOLD", str(DEFAULT_ENABLE_THRESHOLD))
        ),
    )
