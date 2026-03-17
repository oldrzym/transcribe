from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .asr_service import TranscriptionService
from .settings import EnhanceAudioMode, get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

settings = get_settings()
service = TranscriptionService(settings)

app = FastAPI(
    title="GigaAM Transcription API",
    description=(
        "Upload an audio file and get a transcription. "
        "Long audio is split into VAD-aware chunks up to the requested length."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    service.load_dependencies()
    logging.getLogger(__name__).info(
        "Starting API on %s:%s with model=%s device=%s default_chunk_seconds=%s default_enhance_audio_mode=%s",
        settings.host,
        settings.port,
        settings.gigaam_model,
        settings.device,
        settings.default_chunk_seconds,
        settings.default_enhance_audio_mode.value,
    )


@app.get("/")
async def root() -> dict:
    return {
        "service": "gigaam-transcription-api",
        "docs": "/docs",
        "health": "/health",
        "transcribe": "/api/transcribe",
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", **service.health()}


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    chunk_seconds: float | None = Form(default=None),
    enhance_audio_mode: EnhanceAudioMode = Form(
        default=settings.default_enhance_audio_mode
    ),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file was uploaded")

    effective_chunk_seconds = (
        settings.default_chunk_seconds
        if chunk_seconds is None
        else float(chunk_seconds)
    )

    if (
        not settings.min_chunk_seconds
        <= effective_chunk_seconds
        <= settings.max_chunk_seconds
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                f"chunk_seconds must be between "
                f"{settings.min_chunk_seconds} and {settings.max_chunk_seconds}"
            ),
        )

    uploads_dir = settings.work_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename).suffix or ".bin"
    upload_path = uploads_dir / f"{uuid.uuid4().hex}{suffix}"

    try:
        with upload_path.open("wb") as output_file:
            shutil.copyfileobj(file.file, output_file)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to store the uploaded file",
        ) from exc
    finally:
        await file.close()

    try:
        result = service.transcribe_file(
            source_path=upload_path,
            chunk_seconds=effective_chunk_seconds,
            enhance_audio_mode=enhance_audio_mode,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {exc}",
        ) from exc
    finally:
        try:
            upload_path.unlink(missing_ok=True)
        except OSError:
            pass

    return {
        "model": settings.gigaam_model,
        "device": settings.device,
        **result.to_dict(),
    }


if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)
