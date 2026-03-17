FROM python:3.11-slim

ARG PRELOAD_GIGAAM_MODEL=v3_e2e_rnnt

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/models/torch \
    GIGAAM_CACHE_DIR=/models/gigaam \
    PRELOAD_GIGAAM_MODEL=${PRELOAD_GIGAAM_MODEL} \
    SILERO_VAD_REPO_DIR=/opt/silero-vad

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

COPY app /app/app

RUN git clone --depth 1 https://github.com/snakers4/silero-vad.git ${SILERO_VAD_REPO_DIR} && \
    mkdir -p /models/torch /models/gigaam && \
    python -c "import os; import gigaam; import torch; torch.hub.load(repo_or_dir=os.environ['SILERO_VAD_REPO_DIR'], model='silero_vad', source='local', force_reload=False, onnx=False); gigaam.load_model(os.environ['PRELOAD_GIGAAM_MODEL'], device='cpu', download_root=os.environ['GIGAAM_CACHE_DIR'])"

VOLUME ["/models", "/work"]

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
