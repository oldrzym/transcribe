import { requestSummary } from "./api.js";
import {
  getDefaultPromptText,
  INITIAL_STATUS,
  LOADING_STATUS
} from "./content.js";

const state = {
  selectedFile: null,
  draftPrompt: "",
  appliedPrompt: "",
  isLoading: false,
  resultText: "",
  resultMeta: [],
  errorText: "",
  copyLabel: "Скопировать"
};

const elements = {
  fileInput: document.querySelector("#file-input"),
  pickFileButton: document.querySelector("#pick-file-button"),
  dropzone: document.querySelector("#dropzone"),
  fileSummary: document.querySelector("#file-summary"),
  promptTextarea: document.querySelector("#prompt-textarea"),
  applyPromptButton: document.querySelector("#apply-prompt-button"),
  promptStatus: document.querySelector("#prompt-status"),
  defaultPromptText: document.querySelector("#default-prompt-text"),
  submitButton: document.querySelector("#submit-button"),
  spinner: document.querySelector("#spinner"),
  statusTitle: document.querySelector("#status-title"),
  statusCopy: document.querySelector("#status-copy"),
  resultBox: document.querySelector("#result-box"),
  metaRow: document.querySelector("#meta-row"),
  copyButton: document.querySelector("#copy-button")
};

function formatFileSize(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "";
  }

  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unitIndex = 0;

  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }

  return `${value.toFixed(value >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function setStatus(title, copy) {
  elements.statusTitle.textContent = title;
  elements.statusCopy.textContent = copy;
}

function renderMeta(metaItems) {
  if (!metaItems.length) {
    elements.metaRow.innerHTML = "";
    return;
  }

  elements.metaRow.innerHTML = metaItems
    .map((item) => `<span class="meta-pill">${escapeHtml(item)}</span>`)
    .join("");
}

function renderResult() {
  if (state.errorText) {
    elements.resultBox.textContent = state.errorText;
    elements.resultBox.classList.remove("is-placeholder");
    elements.resultBox.classList.add("is-error");
    return;
  }

  if (!state.resultText) {
    elements.resultBox.textContent =
      "Summary появится здесь после обработки файла.";
    elements.resultBox.classList.add("is-placeholder");
    elements.resultBox.classList.remove("is-error");
    return;
  }

  elements.resultBox.textContent = state.resultText;
  elements.resultBox.classList.remove("is-placeholder", "is-error");
}

function render() {
  elements.defaultPromptText.textContent = getDefaultPromptText();

  elements.fileSummary.textContent = state.selectedFile
    ? `${state.selectedFile.name} • ${formatFileSize(state.selectedFile.size)}`
    : "Пока ничего не выбрано";

  const hasAppliedPrompt = Boolean(state.appliedPrompt.trim());
  elements.promptStatus.textContent = hasAppliedPrompt
    ? "Кастомный фокус применен"
    : "Сейчас используется дефолтный prompt";

  elements.applyPromptButton.disabled =
    state.isLoading || state.draftPrompt === state.appliedPrompt;
  elements.submitButton.disabled = state.isLoading || !state.selectedFile;
  elements.copyButton.disabled = !state.resultText;
  elements.copyButton.textContent = state.copyLabel;
  elements.spinner.classList.toggle("hidden", !state.isLoading);

  renderMeta(state.resultMeta);
  renderResult();
}

function setSelectedFile(file) {
  state.selectedFile = file;
  state.errorText = "";
  render();
}

function applyPrompt() {
  state.appliedPrompt = state.draftPrompt.trim();
  state.errorText = "";
  render();
}

function mapResultMeta(payload) {
  const items = [];
  if (payload?.prompt_mode) {
    items.push(
      payload.prompt_mode === "custom"
        ? "Фокус: кастомный"
        : "Фокус: дефолтный"
    );
  }
  if (payload?.transcription?.chunks_count) {
    items.push(`ASR-чанков: ${payload.transcription.chunks_count}`);
  }
  if (payload?.summary_chunking?.chunks_count) {
    items.push(`LLM-чанков: ${payload.summary_chunking.chunks_count}`);
  }
  if (payload?.transcription?.audio_duration_sec) {
    items.push(
      `Длительность: ${Math.round(payload.transcription.audio_duration_sec)} сек`
    );
  }
  return items;
}

async function submit() {
  if (!state.selectedFile || state.isLoading) {
    return;
  }

  state.isLoading = true;
  state.errorText = "";
  state.resultText = "";
  state.resultMeta = [];
  state.copyLabel = "Скопировать";
  setStatus(LOADING_STATUS.title, LOADING_STATUS.copy);
  render();

  try {
    const payload = await requestSummary({
      file: state.selectedFile,
      prompt: state.appliedPrompt
    });

    state.resultText = payload.summary?.trim() || "Сервис вернул пустой результат.";
    state.resultMeta = mapResultMeta(payload);
    setStatus(
      "Summary готово",
      "Результат можно проверить, отредактировать вручную или сразу скопировать."
    );
  } catch (error) {
    state.errorText =
      error instanceof Error ? error.message : "Не удалось получить summary.";
    setStatus(
      "Во время обработки возникла ошибка",
      "Проверьте доступность summary API и попробуйте отправить файл еще раз."
    );
  } finally {
    state.isLoading = false;
    render();
  }
}

async function copyResult() {
  if (!state.resultText) {
    return;
  }

  try {
    await navigator.clipboard.writeText(state.resultText);
    state.copyLabel = "Скопировано";
    render();
    window.setTimeout(() => {
      state.copyLabel = "Скопировать";
      render();
    }, 1600);
  } catch (error) {
    state.copyLabel = "Не удалось";
    render();
  }
}

function bindEvents() {
  elements.pickFileButton.addEventListener("click", () => {
    elements.fileInput.click();
  });

  elements.fileInput.addEventListener("change", (event) => {
    const [file] = event.target.files || [];
    if (file) {
      setSelectedFile(file);
    }
  });

  elements.promptTextarea.addEventListener("input", (event) => {
    state.draftPrompt = event.target.value;
    render();
  });

  elements.applyPromptButton.addEventListener("click", applyPrompt);
  elements.submitButton.addEventListener("click", submit);
  elements.copyButton.addEventListener("click", copyResult);

  ["dragenter", "dragover"].forEach((eventName) => {
    elements.dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.dropzone.classList.add("is-dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    elements.dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.dropzone.classList.remove("is-dragover");
    });
  });

  elements.dropzone.addEventListener("drop", (event) => {
    const [file] = event.dataTransfer?.files || [];
    if (file) {
      setSelectedFile(file);
    }
  });
}

function init() {
  elements.promptTextarea.value = "";
  setStatus(INITIAL_STATUS.title, INITIAL_STATUS.copy);
  bindEvents();
  render();
}

init();
