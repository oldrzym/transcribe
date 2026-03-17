from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

# Both GigaAM and Silero VAD are designed around 16 kHz mono speech audio.
SAMPLE_RATE = 16000
# Remove low-frequency rumble below typical speech fundamentals.
HIGHPASS_HZ = 80
# Keep the useful speech band and cut high-frequency hiss.
LOWPASS_HZ = 7600
# Conservative denoise threshold in dB for ffmpeg afftdn.
NOISE_FLOOR_DB = -25
# Analyze loudness over 150 ms windows for stable normalization.
DYNANORM_FRAME_MS = 150
# Cap adaptive gain to avoid over-amplifying noise.
DYNANORM_MAX_GAIN_DB = 15
ENHANCEMENT_FILTERS = (
    f"highpass=f={HIGHPASS_HZ},"
    f"lowpass=f={LOWPASS_HZ},"
    f"afftdn=nf={NOISE_FLOOR_DB},"
    f"dynaudnorm=f={DYNANORM_FRAME_MS}:g={DYNANORM_MAX_GAIN_DB}"
)

logger = logging.getLogger(__name__)


def convert_audio_for_asr(src: Path, dst: Path, enhance_audio: bool) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)

    # `-vn` drops video stream, `-ac 1` forces mono, `-ar 16000` resamples for ASR.
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
    ]

    if enhance_audio:
        cmd.extend(["-af", ENHANCEMENT_FILTERS])

    cmd.extend(["-f", "wav", str(dst)])

    logger.info("Converting audio with enhance_audio=%s", enhance_audio)
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.error("ffmpeg failed: %s", exc.stderr)
        raise RuntimeError("ffmpeg failed to process the uploaded audio") from exc

    return dst


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        # Downmix multi-channel audio because the ASR pipeline expects mono input.
        audio = np.mean(audio, axis=1).astype("float32")
    return audio, sample_rate


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate)
    return path


def audio_duration_seconds(path: Path) -> float:
    info = sf.info(str(path))
    if not info.samplerate:
        return 0.0
    return float(info.frames) / float(info.samplerate)
