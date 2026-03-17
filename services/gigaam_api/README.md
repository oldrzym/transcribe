# GigaAM Transcription API

This service provides a single FastAPI endpoint for speech transcription with GigaAM.

## What it does

- accepts an uploaded audio file
- automatically assesses recording quality before transcription
- supports `off`, `on`, and `auto` enhancement modes
- splits long audio into chunks
- returns the full transcription and chunk-level timestamps

## Why the default chunk size is 22 seconds

GigaAM documents `.transcribe()` as suitable only for audio up to 25 seconds. In the library source, the long-form helper uses `max_duration=22.0` by default. This project follows that safer default and still allows values up to 25 seconds.

## API

### `POST /api/transcribe`

Multipart form fields:

- `file`: audio file
- `chunk_seconds`: optional float, default `22.0`, allowed range `1.0..25.0`
- `enhance_audio_mode`: optional string, `off | on | auto`, default `auto`

Example:

```bash
curl -X POST "http://localhost:8080/api/transcribe" \
  -F "file=@sample.mp3" \
  -F "chunk_seconds=22" \
  -F "enhance_audio_mode=auto"
```

Response fields:

- `text`: merged transcription
- `audio_duration_sec`: source duration after conversion
- `chunk_seconds`: effective chunk size
- `chunking_mode`: `single`, `vad`, or `fixed`
- `enhance_audio_mode`: requested enhancement mode
- `enhance_audio_applied`: indicates whether enhancement was actually applied
- `quality_assessment`: automatic quality evaluation with metrics, decision, and reasons
- `chunks_count`: number of transcribed chunks
- `chunks`: chunk list with `start_sec`, `end_sec`, `duration_sec`, `text`
- `warnings`: fallback warnings

## Model download behavior

- GigaAM and Silero VAD are downloaded during `docker build`.
- The selected model version is defined once in the root `.env` as `GIGAAM_MODEL` and is reused both for Docker build and for runtime configuration.
- The model cache is stored in `/models`, which is mounted as a Docker volume in `docker-compose.yml`.
- The application loads both dependencies on startup, so the first API request does not trigger downloads.
- If the required model files are missing at runtime, the service fails fast on startup instead of downloading anything on demand.

## Run with Docker Compose

```bash
docker compose up --build
```

The API will be available at `http://localhost:8080` and Swagger UI at `http://localhost:8080/docs`.

## Notes

- default model: `v3_e2e_rnnt`
- default enhancement mode: `auto`
- default runtime: GPU (`APP_DEVICE=cuda`)
- `NVIDIA_VISIBLE_DEVICES` is configured from the root `.env` via `GIGAAM_GPU_DEVICE`
- if `gigaam-api` and `vllm-qwen` point to the same GPU, they will compete for the same VRAM and compute resources
