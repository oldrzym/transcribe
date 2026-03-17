from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .settings import Settings
from .vad_silero import VadSegment

# Small positive value to avoid log10(0) on silent audio.
LOG_EPSILON = 1e-8
# Treat samples above 99% of full scale as likely clipping.
CLIPPING_SAMPLE_LEVEL = 0.99
# Report fully silent or empty audio as -120 dBFS instead of negative infinity.
EMPTY_AUDIO_DBFS = -120.0
# Penalty added when speech level is below the configured threshold.
QUIET_SPEECH_SCORE = 1.0
# Base penalty for noise when the estimated SNR is below target.
LOW_SNR_SCORE = 1.0
# Additional SNR gap that marks the recording as materially noisy.
SEVERE_SNR_GAP_DB = 4.0
# Stronger penalty for clearly noisy recordings.
SEVERE_LOW_SNR_SCORE = 1.5
# Clipping usually harms ASR more than moderate background noise.
CLIPPING_SCORE = 1.25
# Score thresholds used to label the overall recording quality.
BORDERLINE_QUALITY_SCORE = 1.0
POOR_QUALITY_SCORE = 2.0
# Response precision for dB-like metrics.
DB_PRECISION = 3
# Response precision for the final heuristic score.
SCORE_PRECISION = 3
# Response precision for ratios such as clipping share.
RATIO_PRECISION = 6


@dataclass
class AudioQualityAssessment:
    quality_label: str
    quality_score: float
    should_enhance: bool
    analysis_mode: str
    reasons: list[str]
    warnings: list[str]
    metrics: dict[str, float | None]

    def to_dict(self) -> dict:
        return asdict(self)


def assess_audio_quality(
    audio: np.ndarray,
    sample_rate: int,
    settings: Settings,
    speech_segments: list[VadSegment] | None = None,
) -> AudioQualityAssessment:
    total_samples = len(audio)
    total_duration = float(total_samples) / float(sample_rate) if sample_rate else 0.0

    rms_dbfs = _dbfs(audio)
    peak_dbfs = _peak_dbfs(audio)
    clipping_ratio = _clipping_ratio(audio)

    speech_ratio: float | None = None
    speech_rms_dbfs: float | None = None
    silence_rms_dbfs: float | None = None
    snr_proxy_db: float | None = None
    warnings: list[str] = []

    if speech_segments is None:
        analysis_mode = "signal"
        warnings.append(
            "VAD was unavailable during quality assessment; signal-level metrics only were used."
        )
    else:
        analysis_mode = "signal+vad"
        speech_ratio = _speech_ratio(speech_segments, total_duration)
        speech_rms_dbfs, silence_rms_dbfs = _speech_and_silence_levels(
            audio=audio,
            sample_rate=sample_rate,
            speech_segments=speech_segments,
        )
        if speech_rms_dbfs is not None and silence_rms_dbfs is not None:
            snr_proxy_db = speech_rms_dbfs - silence_rms_dbfs
        if (
            speech_ratio is not None
            and speech_ratio < settings.auto_quality_low_speech_ratio_threshold
        ):
            warnings.append(
                f"Low speech ratio detected ({speech_ratio:.1%} of the file contains speech)."
            )

    score = 0.0
    reasons: list[str] = []

    level_dbfs = speech_rms_dbfs if speech_rms_dbfs is not None else rms_dbfs
    if level_dbfs < settings.auto_quality_quiet_speech_dbfs:
        score += QUIET_SPEECH_SCORE
        reasons.append(f"Speech level is low ({level_dbfs:.1f} dBFS).")

    if snr_proxy_db is not None and snr_proxy_db < settings.auto_quality_snr_threshold_db:
        deficit = settings.auto_quality_snr_threshold_db - snr_proxy_db
        score += LOW_SNR_SCORE if deficit < SEVERE_SNR_GAP_DB else SEVERE_LOW_SNR_SCORE
        reasons.append(
            f"Estimated speech-to-noise ratio is low ({snr_proxy_db:.1f} dB)."
        )

    if clipping_ratio >= settings.auto_quality_clipping_ratio_threshold:
        score += CLIPPING_SCORE
        reasons.append(f"Clipping artifacts detected ({clipping_ratio:.2%} of samples).")

    if not reasons:
        reasons.append("Signal quality looks acceptable for direct transcription.")

    quality_label = "good"
    if score >= POOR_QUALITY_SCORE:
        quality_label = "poor"
    elif score >= BORDERLINE_QUALITY_SCORE:
        quality_label = "borderline"

    return AudioQualityAssessment(
        quality_label=quality_label,
        quality_score=round(score, SCORE_PRECISION),
        should_enhance=score >= settings.auto_quality_enable_threshold,
        analysis_mode=analysis_mode,
        reasons=reasons,
        warnings=warnings,
        metrics={
            "duration_sec": round(total_duration, DB_PRECISION),
            "rms_dbfs": round(rms_dbfs, DB_PRECISION),
            "peak_dbfs": round(peak_dbfs, DB_PRECISION),
            "clipping_ratio": round(clipping_ratio, RATIO_PRECISION),
            "speech_ratio": _round_or_none(speech_ratio),
            "speech_rms_dbfs": _round_or_none(speech_rms_dbfs),
            "silence_rms_dbfs": _round_or_none(silence_rms_dbfs),
            "snr_proxy_db": _round_or_none(snr_proxy_db),
        },
    )


def _speech_ratio(segments: list[VadSegment], total_duration: float) -> float | None:
    if total_duration <= 0:
        return None
    speech_duration = 0.0
    for segment in segments:
        speech_duration += max(0.0, segment.duration)
    return min(max(speech_duration / total_duration, 0.0), 1.0)


def _speech_and_silence_levels(
    audio: np.ndarray,
    sample_rate: int,
    speech_segments: list[VadSegment],
) -> tuple[float | None, float | None]:
    speech_sq_sum = 0.0
    speech_samples = 0
    silence_sq_sum = 0.0
    silence_samples = 0
    cursor = 0

    for segment in speech_segments:
        start = max(0, min(len(audio), int(segment.start_sec * sample_rate)))
        end = max(0, min(len(audio), int(segment.end_sec * sample_rate)))

        if start > cursor:
            chunk = audio[cursor:start].astype(np.float64, copy=False)
            silence_sq_sum += float(np.square(chunk).sum())
            silence_samples += len(chunk)

        if end > start:
            chunk = audio[start:end].astype(np.float64, copy=False)
            speech_sq_sum += float(np.square(chunk).sum())
            speech_samples += len(chunk)

        cursor = max(cursor, end)

    if cursor < len(audio):
        chunk = audio[cursor:].astype(np.float64, copy=False)
        silence_sq_sum += float(np.square(chunk).sum())
        silence_samples += len(chunk)

    speech_rms_dbfs = _dbfs_from_stats(speech_sq_sum, speech_samples)
    silence_rms_dbfs = _dbfs_from_stats(silence_sq_sum, silence_samples)
    return speech_rms_dbfs, silence_rms_dbfs


def _dbfs(audio: np.ndarray) -> float:
    if len(audio) == 0:
        return EMPTY_AUDIO_DBFS
    data = audio.astype(np.float64, copy=False)
    rms = float(np.sqrt(np.mean(np.square(data))))
    return float(20.0 * np.log10(max(rms, LOG_EPSILON)))


def _peak_dbfs(audio: np.ndarray) -> float:
    if len(audio) == 0:
        return EMPTY_AUDIO_DBFS
    peak = float(np.max(np.abs(audio)))
    return float(20.0 * np.log10(max(peak, LOG_EPSILON)))


def _clipping_ratio(audio: np.ndarray) -> float:
    if len(audio) == 0:
        return 0.0
    return float(np.mean(np.abs(audio) >= CLIPPING_SAMPLE_LEVEL))


def _dbfs_from_stats(square_sum: float, total_samples: int) -> float | None:
    if total_samples <= 0:
        return None
    rms = float(np.sqrt(square_sum / float(total_samples)))
    return float(20.0 * np.log10(max(rms, LOG_EPSILON)))


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, DB_PRECISION)
