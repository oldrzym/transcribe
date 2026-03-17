# vLLM Qwen Service

This service runs an OpenAI-compatible vLLM server with the official `Qwen/Qwen3-32B-AWQ` model.

## Why this model

- `Qwen/Qwen3-32B` has 32.8B parameters and is not a safe target for a single 40 GB GPU in BF16.
- `Qwen/Qwen3-32B-AWQ` is the official 4-bit AWQ quantized variant and is much more realistic on one 40 GB GPU.
- For transcript summarization workloads, the default 8k context is a practical compromise between VRAM usage and prompt capacity.

## Runtime notes

- The model is downloaded during `docker build`, not at first request.
- The service exposes an OpenAI-compatible API on port `8000` inside the container.
- Readiness can be checked through the built-in `GET /health` and `GET /v1/models` endpoints.
- If you want Qwen3 non-thinking mode, pass `chat_template_kwargs={"enable_thinking": false}` in the request body.

## Hardware note

- On a single 40 GB GPU this profile is expected to be workable.
- Around 30 GB VRAM is not a comfortable target for this 32B model in vLLM, even with AWQ, because weights are only part of total memory usage; KV cache and runtime overhead still matter.
- If `vllm-qwen` and `gigaam-api` are assigned to the same GPU, both services can run at the same time in theory, but they will share the same VRAM and compute budget.
- With the current default `--gpu-memory-utilization 0.88`, vLLM will try to reserve most of a 40 GB card, so running GigaAM on that same GPU is not a safe default.
