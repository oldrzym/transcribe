const FALLBACK_DEFAULT_PROMPT_TEXT =
  "Выдели основную суть обсуждения, договоренности, задачи, сроки, риски и открытые вопросы.";

export function getDefaultPromptText() {
  return window.APP_CONFIG?.defaultPromptText || FALLBACK_DEFAULT_PROMPT_TEXT;
}

export const INITIAL_STATUS = {
  title: "Готово к запуску",
  copy:
    "Выберите файл, при необходимости задайте фокус и отправьте запись на обработку."
};

export const LOADING_STATUS = {
  title: "Идет обработка записи",
  copy:
    "Транскрибация и суммаризация могут занять несколько минут, особенно на длинных встречах."
};
