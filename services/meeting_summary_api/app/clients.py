from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from .settings import Settings


class ServiceClientError(RuntimeError):
    pass


class GigaAMClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            timeout=self.settings.http_timeout_seconds,
            connect=30.0,
        )

    async def transcribe_file(
        self,
        file_path: Path,
        original_filename: str,
        asr_chunk_seconds: float,
        enhance_audio_mode: str,
    ) -> dict[str, Any]:
        url = f"{self.settings.gigaam_api_base_url}/api/transcribe"
        data = {
            "chunk_seconds": str(asr_chunk_seconds),
            "enhance_audio_mode": enhance_audio_mode,
        }

        async with httpx.AsyncClient(timeout=self._timeout()) as client:
            with file_path.open("rb") as file_handle:
                files = {
                    "file": (
                        original_filename,
                        file_handle,
                        "application/octet-stream",
                    )
                }
                response = await client.post(url, data=data, files=files)

        if response.status_code >= 400:
            raise ServiceClientError(
                f"GigaAM API request failed with status {response.status_code}: "
                f"{response.text}"
            )

        payload = response.json()
        if not isinstance(payload, dict) or "text" not in payload:
            raise ServiceClientError("GigaAM API returned an unexpected payload")
        return payload

    async def health(self) -> dict[str, Any]:
        url = f"{self.settings.gigaam_api_base_url}/health"
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
            response = await client.get(url)

        if response.status_code >= 400:
            raise ServiceClientError(
                f"GigaAM health check failed with status {response.status_code}"
            )
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("status") != "ok":
            raise ServiceClientError("GigaAM health check returned an unexpected payload")
        return payload


class VllmClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            timeout=self.settings.http_timeout_seconds,
            connect=30.0,
        )

    def _server_base_url(self) -> str:
        if self.settings.vllm_api_base_url.endswith("/v1"):
            return self.settings.vllm_api_base_url[: -len("/v1")]
        return self.settings.vllm_api_base_url

    async def summarize_messages(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
    ) -> str:
        url = f"{self.settings.vllm_api_base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.settings.vllm_model_name,
            "messages": messages,
            "temperature": self.settings.llm_temperature,
            "max_tokens": max_tokens,
        }

        if not self.settings.llm_enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        async with httpx.AsyncClient(timeout=self._timeout()) as client:
            response = await client.post(url, json=payload)

        if response.status_code >= 400:
            raise ServiceClientError(
                f"vLLM request failed with status {response.status_code}: "
                f"{response.text}"
            )

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ServiceClientError("vLLM returned an unexpected payload") from exc

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            return "\n".join(part.strip() for part in text_parts if part.strip())

        raise ServiceClientError("vLLM returned content in an unsupported format")

    async def health(self) -> dict[str, Any]:
        health_url = f"{self._server_base_url()}/health"
        models_url = f"{self.settings.vllm_api_base_url}/models"

        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
            health_response = await client.get(health_url)
            models_response = await client.get(models_url)

        if health_response.status_code >= 400:
            raise ServiceClientError(
                f"vLLM health endpoint failed with status {health_response.status_code}"
            )
        if models_response.status_code >= 400:
            raise ServiceClientError(
                f"vLLM models endpoint failed with status {models_response.status_code}"
            )

        models_payload = models_response.json()
        if not isinstance(models_payload, dict):
            raise ServiceClientError("vLLM models endpoint returned an unexpected payload")

        models = models_payload.get("data")
        if not isinstance(models, list):
            raise ServiceClientError("vLLM models endpoint returned no models list")

        model_ids = [
            item.get("id")
            for item in models
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ]
        if not model_ids:
            raise ServiceClientError("vLLM models endpoint returned an empty model list")

        return {
            "status": "ok",
            "model_ids": model_ids,
        }
