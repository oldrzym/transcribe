# Audio-Summary

Сервисы проекта вынесены по папкам:

- `services/gigaam_api`
- `services/vllm_qwen`
- `services/meeting_summary_api`
- `services/meeting_summary_frontend`

Запуск:

```bash
docker compose up --build
```

Основные настройки:

- compose-level переменные находятся в `.env`
- runtime-настройки сервиса находятся в `services/gigaam_api/service.env`
- runtime-настройки vLLM находятся в `services/vllm_qwen/service.env`
- runtime-настройки summary-сервиса находятся в `services/meeting_summary_api/service.env`
- runtime-настройки фронта находятся в `services/meeting_summary_frontend/service.env`
- назначение GPU для сервисов тоже задаётся в `.env`

Порты по умолчанию:

- `gigaam-api`: `http://localhost:8080`
- `vllm-qwen`: `http://localhost:8001`
- `meeting-summary-api`: `http://localhost:8082`
- `meeting-summary-web`: `http://localhost:8083`
