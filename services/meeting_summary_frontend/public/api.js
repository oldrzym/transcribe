function getApiBaseUrl() {
  const baseUrl = window.APP_CONFIG?.summaryApiBaseUrl;
  if (!baseUrl) {
    throw new Error("Summary API base URL is not configured");
  }
  return baseUrl.replace(/\/$/, "");
}

function buildEndpoint(path) {
  return `${getApiBaseUrl()}${path}`;
}

async function parseError(response) {
  const rawText = await response.text();

  try {
    const payload = JSON.parse(rawText);
    if (payload?.detail) {
      return typeof payload.detail === "string"
        ? payload.detail
        : JSON.stringify(payload.detail);
    }
  } catch (error) {
    // Ignore JSON parsing errors and fall back to plain text.
  }

  return rawText || `Request failed with status ${response.status}`;
}

export async function requestSummary({ file, prompt }) {
  const formData = new FormData();
  formData.append("file", file);

  if (prompt?.trim()) {
    formData.append("prompt", prompt.trim());
  }

  const response = await fetch(buildEndpoint("/api/summarize"), {
    method: "POST",
    body: formData
  });

  if (!response.ok) {
    throw new Error(await parseError(response));
  }

  return response.json();
}
