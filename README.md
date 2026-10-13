# forglemmigej

*Forglemmigej* is Danish for "forget-me-not". This is a self-hosted Danish tutor: a spaced-repetition app that learns how its user forgets, with a full MLOps loop around the memory model (review logging, nightly retraining, a champion/challenger gate, monitoring and drift detection). It runs on a two-node homelab, without a GPU.

> Work in progress. Built as a learning project for MLOps.

## Repository layout

| Path | What |
|---|---|
| `web/` | SvelteKit app in SPA mode (Svelte 5 + TypeScript) |
| `api/` | FastAPI backend (Python, managed with `uv`) |
| `docker-compose.yml` | Local development services (PostgreSQL) |

## Local development

Prerequisites: Docker, Node.js, [uv](https://docs.astral.sh/uv/).

```bash
cp .env.example .env            # then set POSTGRES_PASSWORD
docker compose up -d            # PostgreSQL on 127.0.0.1:5432

cd api && uv run uvicorn forglemmigej_api.app:app --reload   # API on :8000
cd web && npm run dev                                       # web app on :5173, proxies /api to :8000
```
