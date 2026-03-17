from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .settings import EnhanceAudioMode, get_settings
from .summary_service import MeetingSummaryService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

settings = get_settings()
service = MeetingSummaryService(settings)

app = FastAPI(
    title="Meeting Summary API",
    description=(
        "Upload an audio or video file, transcribe it through GigaAM, "
        "split the transcript into central LLM chunks with neighbor context, "
        "and get a final summary."
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
    await service.wait_for_dependencies()
    logging.getLogger(__name__).info(
        "Starting Meeting Summary API on %s:%s with gigaam=%s vllm=%s",
        settings.host,
        settings.port,
        settings.gigaam_api_base_url,
        settings.vllm_api_base_url,
    )


@app.get("/")
async def root() -> dict:
    return {
        "service": "meeting-summary-api",
        "docs": "/docs",
        "health": "/health",
        "summarize": "/api/summarize",
    }


@app.get("/health")
async def health() -> dict:
    try:
        return await service.health()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/summarize")
async def summarize(
    file: UploadFile = File(...),
    prompt: str | None = Form(default=None),
    asr_chunk_seconds: float | None = Form(default=None),
    enhance_audio_mode: EnhanceAudioMode = Form(
        default=settings.default_enhance_audio_mode
    ),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file was uploaded")

    effective_asr_chunk_seconds = (
        settings.default_asr_chunk_seconds
        if asr_chunk_seconds is None
        else float(asr_chunk_seconds)
    )

    if (
        not settings.min_asr_chunk_seconds
        <= effective_asr_chunk_seconds
        <= settings.max_asr_chunk_seconds
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                f"asr_chunk_seconds must be between "
                f"{settings.min_asr_chunk_seconds} and {settings.max_asr_chunk_seconds}"
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
        result = await service.summarize_file(
            file_path=upload_path,
            original_filename=file.filename,
            prompt=prompt,
            asr_chunk_seconds=effective_asr_chunk_seconds,
            enhance_audio_mode=enhance_audio_mode,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Summarization failed: {exc}",
        ) from exc
    finally:
        try:
            upload_path.unlink(missing_ok=True)
        except OSError:
            pass

    return result


if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)
