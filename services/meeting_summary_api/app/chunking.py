from __future__ import annotations

import re
from dataclasses import dataclass


WHITESPACE_PATTERN = re.compile(r"\s+")
# Prefer paragraph boundaries before sentence-level fallback.
PARAGRAPH_SPLIT_PATTERN = re.compile(r"\n\s*\n+")
# Sentence breaks are an acceptable fallback when paragraphs are too large.
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class TranscriptSegment:
    index: int
    text: str
    source_chunk_indices: list[int]
    start_sec: float | None = None
    end_sec: float | None = None

    @property
    def word_count(self) -> int:
        return count_words(self.text)


@dataclass(frozen=True)
class SummaryChunk:
    index: int
    text: str
    left_context_text: str
    right_context_text: str
    source_chunk_indices: list[int]
    word_count: int
    char_count: int
    start_sec: float | None
    end_sec: float | None


def count_words(text: str) -> int:
    normalized = WHITESPACE_PATTERN.sub(" ", text).strip()
    if not normalized:
        return 0
    return len(normalized.split(" "))


def normalize_text(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def build_summary_chunks(
    transcript_text: str,
    transcript_chunks: list[dict],
    *,
    max_words: int,
    max_chars: int,
    context_words: int,
) -> list[SummaryChunk]:
    segments = _prepare_segments(
        transcript_text=transcript_text,
        transcript_chunks=transcript_chunks,
        max_words=max_words,
        max_chars=max_chars,
    )
    if not segments:
        return []

    groups = _build_non_overlapping_groups(
        segments=segments,
        max_words=max_words,
        max_chars=max_chars,
    )
    base_chunks: list[SummaryChunk] = []
    for index, group in enumerate(groups, start=1):
        merged_text = "\n".join(segment.text for segment in group if segment.text).strip()
        if not merged_text:
            continue

        source_chunk_indices = _deduplicate_in_order(
            source_index
            for segment in group
            for source_index in segment.source_chunk_indices
        )
        start_sec = next(
            (segment.start_sec for segment in group if segment.start_sec is not None),
            None,
        )
        end_sec = next(
            (
                segment.end_sec
                for segment in reversed(group)
                if segment.end_sec is not None
            ),
            None,
        )
        base_chunks.append(
            SummaryChunk(
                index=index,
                text=merged_text,
                left_context_text="",
                right_context_text="",
                source_chunk_indices=source_chunk_indices,
                word_count=count_words(merged_text),
                char_count=len(merged_text),
                start_sec=start_sec,
                end_sec=end_sec,
            )
        )

    return _attach_neighbor_context(base_chunks, context_words=context_words)


def _prepare_segments(
    transcript_text: str,
    transcript_chunks: list[dict],
    *,
    max_words: int,
    max_chars: int,
) -> list[TranscriptSegment]:
    raw_segments = _segments_from_gigaam_chunks(transcript_chunks)
    if not raw_segments:
        raw_segments = _segments_from_text_fallback(
            transcript_text,
            max_words=max_words,
            max_chars=max_chars,
        )

    prepared: list[TranscriptSegment] = []
    synthetic_index = 1
    for segment in raw_segments:
        parts = _split_text_block(
            segment.text,
            max_words=max_words,
            max_chars=max_chars,
        )
        if len(parts) <= 1:
            prepared.append(
                TranscriptSegment(
                    index=synthetic_index,
                    text=segment.text,
                    source_chunk_indices=segment.source_chunk_indices.copy(),
                    start_sec=segment.start_sec,
                    end_sec=segment.end_sec,
                )
            )
            synthetic_index += 1
            continue

        for part in parts:
            prepared.append(
                TranscriptSegment(
                    index=synthetic_index,
                    text=part,
                    source_chunk_indices=segment.source_chunk_indices.copy(),
                    start_sec=segment.start_sec,
                    end_sec=segment.end_sec,
                )
            )
            synthetic_index += 1

    return prepared


def _segments_from_gigaam_chunks(transcript_chunks: list[dict]) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    for item in transcript_chunks:
        if not isinstance(item, dict):
            continue

        text = normalize_text(str(item.get("text", "")))
        if not text:
            continue

        chunk_index = int(item.get("index", len(segments) + 1))
        segments.append(
            TranscriptSegment(
                index=chunk_index,
                text=text,
                source_chunk_indices=[chunk_index],
                start_sec=_as_optional_float(item.get("start_sec")),
                end_sec=_as_optional_float(item.get("end_sec")),
            )
        )
    return segments


def _segments_from_text_fallback(
    transcript_text: str,
    *,
    max_words: int,
    max_chars: int,
) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    blocks = _split_text_block(
        transcript_text,
        max_words=max(1, max_words // 2),
        max_chars=max(1, max_chars // 2),
    )
    for index, block in enumerate(blocks, start=1):
        segments.append(
            TranscriptSegment(
                index=index,
                text=block,
                source_chunk_indices=[],
            )
        )
    return segments


def _split_text_block(
    text: str,
    *,
    max_words: int,
    max_chars: int,
) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []

    if count_words(normalized) <= max_words and len(normalized) <= max_chars:
        return [normalized]

    parts = _smart_split(normalized)
    if len(parts) == 1:
        return _split_by_words(normalized, max_words=max_words, max_chars=max_chars)

    result: list[str] = []
    current = ""

    for part in parts:
        candidate = f"{current}\n\n{part}".strip() if current else part
        if count_words(candidate) <= max_words and len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            result.append(current)
        if count_words(part) <= max_words and len(part) <= max_chars:
            current = part
            continue

        result.extend(_split_by_words(part, max_words=max_words, max_chars=max_chars))
        current = ""

    if current:
        result.append(current)

    return [item for item in result if item.strip()]


def _smart_split(text: str) -> list[str]:
    paragraphs = [
        paragraph.strip()
        for paragraph in PARAGRAPH_SPLIT_PATTERN.split(text)
        if paragraph.strip()
    ]
    if len(paragraphs) > 1:
        return paragraphs

    sentences = [
        sentence.strip()
        for sentence in SENTENCE_SPLIT_PATTERN.split(text)
        if sentence.strip()
    ]
    if len(sentences) > 1:
        return sentences

    return [text.strip()]


def _split_by_words(text: str, *, max_words: int, max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    parts: list[str] = []
    current_words: list[str] = []

    for word in words:
        candidate_words = current_words + [word]
        candidate_text = " ".join(candidate_words)
        if len(candidate_words) <= max_words and len(candidate_text) <= max_chars:
            current_words = candidate_words
            continue

        if current_words:
            parts.append(" ".join(current_words))
        current_words = [word]

    if current_words:
        parts.append(" ".join(current_words))

    return parts


def _build_non_overlapping_groups(
    *,
    segments: list[TranscriptSegment],
    max_words: int,
    max_chars: int,
) -> list[list[TranscriptSegment]]:
    groups: list[list[TranscriptSegment]] = []
    current_group: list[TranscriptSegment] = []

    for segment in segments:
        if current_group and _would_exceed_limits(
            current_group,
            segment,
            max_words=max_words,
            max_chars=max_chars,
        ):
            groups.append(current_group)
            current_group = []

        current_group.append(segment)

    if current_group:
        groups.append(current_group)

    return groups


def _would_exceed_limits(
    current_group: list[TranscriptSegment],
    segment: TranscriptSegment,
    *,
    max_words: int,
    max_chars: int,
) -> bool:
    current_text = "\n".join(item.text for item in current_group if item.text).strip()
    candidate_text = (
        f"{current_text}\n{segment.text}".strip() if current_text else segment.text
    )
    return count_words(candidate_text) > max_words or len(candidate_text) > max_chars


def _attach_neighbor_context(
    chunks: list[SummaryChunk],
    *,
    context_words: int,
) -> list[SummaryChunk]:
    if context_words <= 0:
        return chunks

    enriched_chunks: list[SummaryChunk] = []
    for index, chunk in enumerate(chunks):
        previous_chunk = chunks[index - 1] if index > 0 else None
        next_chunk = chunks[index + 1] if index + 1 < len(chunks) else None
        left_context_text = (
            _take_tail_words(previous_chunk.text, context_words)
            if previous_chunk is not None
            else ""
        )
        right_context_text = (
            _take_head_words(next_chunk.text, context_words)
            if next_chunk is not None
            else ""
        )
        enriched_chunks.append(
            SummaryChunk(
                index=chunk.index,
                text=chunk.text,
                left_context_text=left_context_text,
                right_context_text=right_context_text,
                source_chunk_indices=chunk.source_chunk_indices.copy(),
                word_count=chunk.word_count,
                char_count=chunk.char_count,
                start_sec=chunk.start_sec,
                end_sec=chunk.end_sec,
            )
        )

    return enriched_chunks


def _deduplicate_in_order(values) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _as_optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _take_head_words(text: str, max_words: int) -> str:
    words = text.split()
    if max_words <= 0 or len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()


def _take_tail_words(text: str, max_words: int) -> str:
    words = text.split()
    if max_words <= 0 or len(words) <= max_words:
        return text.strip()
    return " ".join(words[-max_words:]).strip()
