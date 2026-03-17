FROM vllm/vllm-openai:v0.12.0

ARG VLLM_MODEL_REPO=Qwen/Qwen3-32B-AWQ
ARG VLLM_LOCAL_MODEL_DIR=/opt/models/qwen3-32b-awq

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    VLLM_LOCAL_MODEL_DIR=${VLLM_LOCAL_MODEL_DIR}

# Download the model at image build time so the runtime container stays network-free.
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${VLLM_MODEL_REPO}', local_dir='${VLLM_LOCAL_MODEL_DIR}', local_dir_use_symlinks=False)"

VOLUME ["/root/.cache/vllm"]

EXPOSE 8000
