# Meeting Summary Frontend

Static frontend container for the meeting summary flow.

## What it does

- lets the user drag and drop or choose an audio/video file
- lets the user type and apply a custom focus prompt
- shows the current default prompt in a muted helper block
- sends the file and the applied prompt to `meeting-summary-api`
- shows a loading spinner while the request is running
- renders the final summary and allows copying it with one click

## Runtime configuration

The frontend is served by `nginx` and reads the summary API URL at container startup.

Runtime variable:

- `SUMMARY_API_BASE_URL`: public browser-facing base URL of `meeting-summary-api`
- `DEFAULT_PROMPT_TEXT`: text shown in the muted default-prompt helper block

Example:

```bash
SUMMARY_API_BASE_URL=http://localhost:8082
DEFAULT_PROMPT_TEXT=Выдели основную суть обсуждения, договоренности, задачи, сроки, риски и открытые вопросы.
```

## Run with Docker Compose

```bash
docker compose up --build
```

The UI will be available at `http://localhost:8083`.
