# Backend

Weapon detection and sensor orchestration API.

## Layout

- `apps/` — thin runnable entrypoints (API server, capture, infer worker)
- `api/` — FastAPI HTTP layer
- `services/` — business logic with no HTTP framework ownership
- `runtime/` — long-lived sensor/infer runners
- `media/` — frame capture, encode, IPC
- `weapon_ai/` — ML inference backbone
- `utils/` — shared helpers
- `config/` — static settings and model profiles
- `artifacts/` — runtime outputs (gitignored)

## Quick start

```bash
cp .env.example .env
make install
make api          # start FastAPI API
make infer        # start infer worker (optional)
make test
```
