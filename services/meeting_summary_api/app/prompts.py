from __future__ import annotations

from .chunking import SummaryChunk


CHUNK_SYSTEM_PROMPT = (
    "Ты помогаешь готовить summary по расшифровкам встреч. "
    "Работай только по тексту, не выдумывай факты и пиши по-русски."
)

FINAL_SYSTEM_PROMPT = (
    "Ты собираешь итоговое summary встречи из промежуточных summary. "
    "Удали повторы, объедини совпадающие пункты и оставь только подтвержденную текстом информацию."
)


def build_chunk_messages(
    chunk: SummaryChunk,
    custom_prompt: str | None,
) -> list[dict[str, str]]:
    focus_block = _build_focus_block(custom_prompt)
    time_block = _build_time_block(chunk)
    context_sections = _build_context_sections(chunk)
    user_prompt = (
        "Ниже даны блоки текста одной встречи.\n"
        f"{time_block}"
        "Используй соседние блоки только для понимания контекста:\n"
        "- кто говорит;\n"
        "- к чему относятся местоимения и ссылки;\n"
        "- чем заканчиваются или продолжаются мысли.\n"
        "Извлекай факты, договоренности, задачи, сроки, риски и просьбы только из центрального фрагмента.\n"
        "Не включай в ответ факты, которые есть только в соседних блоках.\n"
        "Сделай краткое структурированное summary только по центральному фрагменту.\n"
        "В ответе выдели:\n"
        "- основную суть обсуждения;\n"
        "- кто что попросил, пообещал или подтвердил;\n"
        "- задачи, договоренности и следующие шаги;\n"
        "- сроки, риски и открытые вопросы, если они есть.\n"
        "Если каких-то данных нет, не додумывай их.\n"
        f"{focus_block}"
        f"\n{context_sections}"
    )
    return [
        {"role": "system", "content": CHUNK_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_final_messages(
    partial_summaries: list[dict[str, str | int | float | None]],
    custom_prompt: str | None,
) -> list[dict[str, str]]:
    focus_block = _build_focus_block(custom_prompt)
    parts_text = "\n\n".join(
        _render_partial_summary(item)
        for item in partial_summaries
        if str(item.get("summary", "")).strip()
    )
    user_prompt = (
        "Ниже приведены промежуточные summary частей одной встречи.\n"
        "Собери из них единое итоговое summary без повторов.\n"
        "Структура ответа:\n"
        "- общий контекст;\n"
        "- ключевые решения и договоренности;\n"
        "- задачи и ответственные, если они названы;\n"
        "- сроки;\n"
        "- риски и открытые вопросы.\n"
        "Если в исходных summary чего-то нет, не придумывай это.\n"
        f"{focus_block}"
        f"\nПромежуточные summary:\n{parts_text}"
    )
    return [
        {"role": "system", "content": FINAL_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _build_focus_block(custom_prompt: str | None) -> str:
    if not custom_prompt or not custom_prompt.strip():
        return ""
    return (
        "\nДополнительный фокус пользователя:\n"
        f"{custom_prompt.strip()}\n"
    )


def _build_time_block(chunk: SummaryChunk) -> str:
    if chunk.start_sec is None or chunk.end_sec is None:
        return ""
    return f"Таймкод фрагмента: {chunk.start_sec:.1f}s - {chunk.end_sec:.1f}s.\n"


def _build_context_sections(chunk: SummaryChunk) -> str:
    sections: list[str] = []
    if chunk.left_context_text:
        sections.append(f"Левый контекст:\n{chunk.left_context_text}")
    sections.append(f"Центральный фрагмент {chunk.index}:\n{chunk.text}")
    if chunk.right_context_text:
        sections.append(f"Правый контекст:\n{chunk.right_context_text}")
    return "\n\n".join(sections)


def _render_partial_summary(item: dict[str, str | int | float | None]) -> str:
    index = item.get("index")
    start_sec = item.get("start_sec")
    end_sec = item.get("end_sec")
    summary = str(item.get("summary", "")).strip()

    header = f"Часть {index}"
    if isinstance(start_sec, (int, float)) and isinstance(end_sec, (int, float)):
        header += f" ({float(start_sec):.1f}s - {float(end_sec):.1f}s)"
    return f"{header}\n{summary}"
