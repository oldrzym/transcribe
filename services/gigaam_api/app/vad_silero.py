from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

# Keep the same minimum useful segment size in all chunking code paths.
MIN_SEGMENT_SECONDS = 0.2
# Silero timestamps are returned in samples for 16 kHz audio.
VAD_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class VadSegment:
    start_sec: float
    end_sec: float

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


class SileroVad:
    def __init__(self, device: str = "cpu", repo_dir: str = "/opt/silero-vad"):
        if device.lower().startswith("cuda"):
            logger.warning("Silero VAD is loaded on CPU for compatibility.")
            device = "cpu"

        self.device = device
        self.repo_dir = repo_dir
        self.model, self.utils = torch.hub.load(
            repo_or_dir=self.repo_dir,
            model="silero_vad",
            source="local",
            force_reload=False,
            onnx=False,
        )
        self.model.to(self.device)
        (
            self.get_speech_timestamps,
            _save_audio,
            self.read_audio,
            _vad_iterator,
            _collect_chunks,
        ) = self.utils

    def get_segments(self, wav_path: str) -> list[VadSegment]:
        wav = self.read_audio(wav_path, sampling_rate=VAD_SAMPLE_RATE)
        timestamps = self.get_speech_timestamps(
            wav,
            self.model,
            sampling_rate=VAD_SAMPLE_RATE,
        )

        segments: list[VadSegment] = []
        for item in timestamps:
            start_sec = item["start"] / float(VAD_SAMPLE_RATE)
            end_sec = item["end"] / float(VAD_SAMPLE_RATE)
            if end_sec > start_sec:
                segments.append(VadSegment(start_sec=start_sec, end_sec=end_sec))

        return segments


def split_to_max_len(segments: list[VadSegment], max_len: float) -> list[VadSegment]:
    result: list[VadSegment] = []
    for segment in segments:
        cursor = segment.start_sec
        while cursor < segment.end_sec:
            next_stop = min(cursor + max_len, segment.end_sec)
            if next_stop - cursor >= MIN_SEGMENT_SECONDS:
                result.append(VadSegment(start_sec=cursor, end_sec=next_stop))
            cursor = next_stop
    return result


def build_fixed_segments(total_duration: float, max_len: float) -> list[VadSegment]:
    result: list[VadSegment] = []
    cursor = 0.0
    while cursor < total_duration:
        next_stop = min(cursor + max_len, total_duration)
        if next_stop - cursor >= MIN_SEGMENT_SECONDS:
            result.append(VadSegment(start_sec=cursor, end_sec=next_stop))
        cursor = next_stop
    return result
