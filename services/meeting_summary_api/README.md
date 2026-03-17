# Meeting Summary API

This service orchestrates the full summarization pipeline:

1. uploads a media file
2. sends it to `gigaam-api` for transcription
3. groups transcript segments into central LLM chunks
4. adds compact left and right neighbor context around each central chunk
5. summarizes each chunk sequentially through `vllm-qwen`
6. merges all partial summaries into one final summary

On startup, the service waits until both upstream dependencies are really ready:

- `gigaam-api` responds from `/health`
- `vllm-qwen` responds from `/health` and exposes at least one model through `/v1/models`

## Why chunk by central ASR-based chunks with neighbor context

For long meetings, especially 1-2 hours, it is usually better to chunk by groups of ASR segments rather than by raw character windows:

- speech boundaries are already reflected in the GigaAM transcript chunks
- we avoid cutting decisions and task statements in the middle of a phrase
- the model sees a compact piece of the previous and next chunk for context
- the prompt explicitly tells the model to summarize only the central chunk

The service still has a text-based fallback if upstream chunk metadata is unavailable.

## API

### `POST /api/summarize`

Multipart form fields:

- `file`: source audio or video file
- `prompt`: optional custom focus prompt; when omitted, the default summary prompt is used
- `asr_chunk_seconds`: optional float passed through to `gigaam-api`, default `22.0`
- `enhance_audio_mode`: optional `off | on | auto`, default `auto`

Example:

```bash
curl -X POST "http://localhost:8082/api/summarize" \
  -F "file=@meeting.mp4" \
  -F "prompt=Сделай акцент на поручениях и рисках" \
  -F "asr_chunk_seconds=22" \
  -F "enhance_audio_mode=auto"
```

Response fields:

- `summary`: final merged summary
- `prompt_mode`: `default` or `custom`
- `transcription`: upstream transcription metadata and full transcript text
- `summary_chunking`: applied LLM chunking settings
- `chunk_summaries`: sequential partial summaries for each LLM chunk

## Run with Docker Compose

```bash
docker compose up --build
```

The API will be available at `http://localhost:8082` and Swagger UI at `http://localhost:8082/docs`.
