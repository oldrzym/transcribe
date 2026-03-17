from __future__ import annotations

import asyncio
import logging
from time import monotonic
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .chunking import build_summary_chunks
from .clients import GigaAMClient, VllmClient
from .prompts import build_chunk_messages, build_final_messages
from .settings import EnhanceAudioMode, Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PartialSummary:
    index: int
    start_sec: float | None
    end_sec: float | None
    source_chunk_indices: list[int]
    word_count: int
    char_count: int
    summary: str


class MeetingSummaryService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.gigaam_client = GigaAMClient(settings)
        self.vllm_client = VllmClient(settings)

    async def health(self) -> dict[str, Any]:
        gigaam_health = await self.gigaam_client.health()
        vllm_health = await self.vllm_client.health()
        return {
            "status": "ok",
            "gigaam": gigaam_health,
            "vllm": vllm_health,
        }

    async def wait_for_dependencies(self) -> dict[str, Any]:
        deadline = monotonic() + self.settings.dependency_wait_timeout_seconds
        last_error: Exception | None = None
        attempt = 0

        while monotonic() < deadline:
            attempt += 1
            try:
                health_payload = await self.health()
                logger.info(
                    "Dependencies are ready after %s attempt(s): gigaam=%s vllm=%s",
                    attempt,
                    self.settings.gigaam_api_base_url,
                    self.settings.vllm_api_base_url,
                )
                return health_payload
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Waiting for dependencies (attempt %s): %s",
                    attempt,
                    exc,
                )
                await asyncio.sleep(self.settings.dependency_wait_interval_seconds)

        raise RuntimeError(
            "Dependent services did not become ready in time. "
            f"Last error: {last_error}"
        )

    async def summarize_file(
        self,
        *,
        file_path: Path,
        original_filename: str,
        prompt: str | None,
        asr_chunk_seconds: float,
        enhance_audio_mode: EnhanceAudioMode,
    ) -> dict[str, Any]:
        transcription = await self.gigaam_client.transcribe_file(
            file_path=file_path,
            original_filename=original_filename,
            asr_chunk_seconds=asr_chunk_seconds,
            enhance_audio_mode=enhance_audio_mode.value,
        )

        transcript_text = str(transcription.get("text", "")).strip()
        if not transcript_text:
            raise RuntimeError("GigaAM returned an empty transcription")

        transcript_chunks = transcription.get("chunks")
        if not isinstance(transcript_chunks, list):
            transcript_chunks = []

        summary_chunks = build_summary_chunks(
            transcript_text=transcript_text,
            transcript_chunks=transcript_chunks,
            max_words=self.settings.summary_max_words,
            max_chars=self.settings.summary_max_chars,
            context_words=self.settings.summary_context_words,
        )
        if not summary_chunks:
            raise RuntimeError("Failed to prepare LLM chunks from the transcription")

        partial_summaries: list[PartialSummary] = []
        for chunk in summary_chunks:
            summary_text = await self.vllm_client.summarize_messages(
                build_chunk_messages(chunk, prompt),
                max_tokens=self.settings.llm_chunk_max_tokens,
            )
            partial_summaries.append(
                PartialSummary(
                    index=chunk.index,
                    start_sec=chunk.start_sec,
                    end_sec=chunk.end_sec,
                    source_chunk_indices=chunk.source_chunk_indices,
                    word_count=chunk.word_count,
                    char_count=chunk.char_count,
                    summary=summary_text,
                )
            )

        if not any(item.summary.strip() for item in partial_summaries):
            raise RuntimeError("LLM returned empty summaries for all transcript chunks")

        final_summary = await self.vllm_client.summarize_messages(
            build_final_messages(
                [asdict(item) for item in partial_summaries],
                prompt,
            ),
            max_tokens=self.settings.llm_final_max_tokens,
        )

        return {
            "summary": final_summary,
            "prompt_mode": "custom" if prompt and prompt.strip() else "default",
            "transcription": {
                "text": transcript_text,
                "audio_duration_sec": transcription.get("audio_duration_sec"),
                "chunks_count": transcription.get("chunks_count"),
                "chunking_mode": transcription.get("chunking_mode"),
                "enhance_audio_mode": transcription.get("enhance_audio_mode"),
                "enhance_audio_applied": transcription.get("enhance_audio_applied"),
                "quality_assessment": transcription.get("quality_assessment"),
                "warnings": transcription.get("warnings") or [],
            },
            "summary_chunking": {
                "strategy": "central_chunk_with_neighbor_context",
                "chunks_count": len(summary_chunks),
                "max_words": self.settings.summary_max_words,
                "max_chars": self.settings.summary_max_chars,
                "context_words": self.settings.summary_context_words,
            },
            "chunk_summaries": [asdict(item) for item in partial_summaries],
        }
