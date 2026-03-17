from __future__ import annotations

import logging
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import gigaam
import numpy as np

from .audio_io import (
    convert_audio_for_asr,
    read_wav,
    write_wav,
)
from .quality import AudioQualityAssessment, assess_audio_quality
from .settings import EnhanceAudioMode, Settings
from .vad_silero import SileroVad, VadSegment, build_fixed_segments, split_to_max_len

logger = logging.getLogger(__name__)

# Fragments shorter than 200 ms are usually too small to produce stable ASR output.
MIN_TRANSCRIBABLE_CHUNK_SECONDS = 0.2
# GigaAM supports short aliases that resolve to the latest v3 generation.
SHORT_MODEL_NAMES = {"ctc", "rnnt", "e2e_ctc", "e2e_rnnt", "ssl"}


@dataclass
class TranscriptChunk:
    index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    text: str


@dataclass
class TranscriptionResult:
    text: str
    audio_duration_sec: float
    chunk_seconds: float
    chunking_mode: str
    enhance_audio_mode: str
    enhance_audio_applied: bool
    quality_assessment: AudioQualityAssessment
    chunks: list[TranscriptChunk]
    warnings: list[str]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["chunks_count"] = len(self.chunks)
        return payload


class TranscriptionService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._vad = None

    def load_dependencies(self) -> None:
        self._load_vad()
        self._load_model()

    def health(self) -> dict:
        return {
            "model": self.settings.gigaam_model,
            "device": self.settings.device,
            "default_chunk_seconds": self.settings.default_chunk_seconds,
            "max_chunk_seconds": self.settings.max_chunk_seconds,
            "default_enhance_audio_mode": self.settings.default_enhance_audio_mode.value,
            "model_loaded": self._model is not None,
            "vad_loaded": self._vad is not None,
        }

    def _load_model(self):
        if self._model is None:
            self._assert_model_assets_present()
            logger.info(
                "Loading GigaAM model=%s on device=%s",
                self.settings.gigaam_model,
                self.settings.device,
            )
            self._model = gigaam.load_model(
                self.settings.gigaam_model,
                device=self.settings.device,
                download_root=str(self.settings.model_cache_dir),
            )
        return self._model

    def _load_vad(self) -> SileroVad:
        if self._vad is None:
            logger.info("Loading Silero VAD")
            self._vad = SileroVad(
                device="cpu",
                repo_dir=str(self.settings.silero_vad_repo_dir),
            )
        return self._vad

    def _require_model(self):
        if self._model is None:
            raise RuntimeError("GigaAM model is not initialized")
        return self._model

    def _require_vad(self) -> SileroVad:
        if self._vad is None:
            raise RuntimeError("Silero VAD is not initialized")
        return self._vad

    def _assert_model_assets_present(self) -> None:
        normalized_model_name = self._normalize_model_name(self.settings.gigaam_model)
        expected_paths = [self.settings.model_cache_dir / f"{normalized_model_name}.ckpt"]

        if normalized_model_name == "v1_rnnt" or "e2e" in normalized_model_name:
            expected_paths.append(
                self.settings.model_cache_dir
                / f"{normalized_model_name}_tokenizer.model"
            )

        missing_paths = [path for path in expected_paths if not path.exists()]
        if missing_paths:
            missing_paths_text = ", ".join(str(path) for path in missing_paths)
            raise RuntimeError(
                "Required GigaAM model files are missing from /models. "
                "Rebuild the Docker image or recreate the model volume. "
                f"Missing: {missing_paths_text}"
            )

    def _normalize_model_name(self, model_name: str) -> str:
        if model_name in SHORT_MODEL_NAMES:
            return f"v3_{model_name}"
        return model_name

    def transcribe_file(
        self,
        source_path: Path,
        chunk_seconds: float,
        enhance_audio_mode: EnhanceAudioMode,
    ) -> TranscriptionResult:
        warnings: list[str] = []
        tmp_root = self.settings.work_dir / "tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            raw_wav = tmp_dir / "prepared_raw.wav"
            prepared_wav = raw_wav
            convert_audio_for_asr(source_path, raw_wav, enhance_audio=False)
            quality_assessment = self._assess_quality(raw_wav)

            enhance_audio_applied = False
            if enhance_audio_mode == EnhanceAudioMode.ON:
                enhanced_wav = tmp_dir / "prepared_enhanced.wav"
                convert_audio_for_asr(source_path, enhanced_wav, enhance_audio=True)
                prepared_wav = enhanced_wav
                enhance_audio_applied = True
            elif enhance_audio_mode == EnhanceAudioMode.AUTO:
                if quality_assessment.should_enhance:
                    enhanced_wav = tmp_dir / "prepared_enhanced.wav"
                    try:
                        convert_audio_for_asr(
                            source_path,
                            enhanced_wav,
                            enhance_audio=True,
                        )
                        prepared_wav = enhanced_wav
                        enhance_audio_applied = True
                    except RuntimeError:
                        warnings.append(
                            "Automatic enhancement failed; raw audio was used instead."
                        )

            audio, sample_rate = read_wav(prepared_wav)
            total_duration = float(len(audio)) / float(sample_rate) if sample_rate else 0.0
            if total_duration <= 0:
                raise RuntimeError("Prepared audio is empty")

            model = self._require_model()
            if total_duration <= chunk_seconds:
                text = model.transcribe(str(prepared_wav)).strip()
                chunk = TranscriptChunk(
                    index=1,
                    start_sec=0.0,
                    end_sec=total_duration,
                    duration_sec=total_duration,
                    text=text,
                )
                return TranscriptionResult(
                    text=text,
                    audio_duration_sec=total_duration,
                    chunk_seconds=chunk_seconds,
                    chunking_mode="single",
                    enhance_audio_mode=enhance_audio_mode.value,
                    enhance_audio_applied=enhance_audio_applied,
                    quality_assessment=quality_assessment,
                    chunks=[chunk],
                    warnings=warnings,
                )

            segments, chunking_mode, mode_warnings = self._build_segments(
                prepared_wav=prepared_wav,
                audio=audio,
                sample_rate=sample_rate,
                chunk_seconds=chunk_seconds,
            )
            warnings.extend(mode_warnings)

            chunks: list[TranscriptChunk] = []
            for index, segment in enumerate(segments, start=1):
                chunk_text = self._transcribe_segment(
                    tmp_dir=tmp_dir,
                    audio=audio,
                    sample_rate=sample_rate,
                    segment=segment,
                    index=index,
                    model=model,
                )
                if chunk_text is None:
                    continue

                chunks.append(
                    TranscriptChunk(
                        index=index,
                        start_sec=segment.start_sec,
                        end_sec=segment.end_sec,
                        duration_sec=segment.duration,
                        text=chunk_text,
                    )
                )

            full_text = "\n".join(chunk.text for chunk in chunks if chunk.text).strip()

            return TranscriptionResult(
                text=full_text,
                audio_duration_sec=total_duration,
                chunk_seconds=chunk_seconds,
                chunking_mode=chunking_mode,
                enhance_audio_mode=enhance_audio_mode.value,
                enhance_audio_applied=enhance_audio_applied,
                quality_assessment=quality_assessment,
                chunks=chunks,
                warnings=warnings,
            )

    def _assess_quality(self, wav_path: Path) -> AudioQualityAssessment:
        audio, sample_rate = read_wav(wav_path)
        speech_segments: list[VadSegment] | None
        try:
            speech_segments = self._require_vad().get_segments(str(wav_path))
        except Exception as exc:
            logger.warning("Quality assessment VAD failed: %s", exc)
            speech_segments = None
        return assess_audio_quality(
            audio=audio,
            sample_rate=sample_rate,
            settings=self.settings,
            speech_segments=speech_segments,
        )

    def _build_segments(
        self,
        prepared_wav: Path,
        audio: np.ndarray,
        sample_rate: int,
        chunk_seconds: float,
    ) -> tuple[list[VadSegment], str, list[str]]:
        total_duration = float(len(audio)) / float(sample_rate)

        try:
            speech_segments = self._require_vad().get_segments(str(prepared_wav))
        except Exception as exc:
            logger.warning("VAD failed, falling back to fixed chunks: %s", exc)
            return (
                build_fixed_segments(total_duration=total_duration, max_len=chunk_seconds),
                "fixed",
                ["VAD failed, fixed-duration chunking was used instead."],
            )

        if not speech_segments:
            logger.warning("VAD found no speech, falling back to fixed chunks")
            return (
                build_fixed_segments(total_duration=total_duration, max_len=chunk_seconds),
                "fixed",
                ["VAD found no speech, fixed-duration chunking was used instead."],
            )

        return split_to_max_len(speech_segments, max_len=chunk_seconds), "vad", []

    def _transcribe_segment(
        self,
        tmp_dir: Path,
        audio: np.ndarray,
        sample_rate: int,
        segment: VadSegment,
        index: int,
        model,
    ) -> str | None:
        start_sample = max(0, int(segment.start_sec * sample_rate))
        end_sample = min(len(audio), int(segment.end_sec * sample_rate))
        chunk_audio = audio[start_sample:end_sample]

        if len(chunk_audio) < int(MIN_TRANSCRIBABLE_CHUNK_SECONDS * sample_rate):
            return None

        segment_path = tmp_dir / f"segment_{index:06d}.wav"
        write_wav(segment_path, chunk_audio, sample_rate)
        return model.transcribe(str(segment_path)).strip()
