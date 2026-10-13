# forglemmigej

*Forglemmigej* is Danish for "forget-me-not". This is a self-hosted Danish tutor: a spaced-repetition app that learns how its user forgets, with a full MLOps loop around the memory model (review logging, nightly retraining, a champion/challenger gate, monitoring and drift detection). It runs on a two-node homelab, without a GPU.

> Work in progress. Built as a learning project for MLOps.

## Repository layout

| Path | What |
|---|---|
| `web/` | SvelteKit app in SPA mode (Svelte 5 + TypeScript) |
| `api/` | FastAPI backend (Python, managed with `uv`) |
| `data/curation/` | Hand-maintained word corrections and survival phrases |
| `docker-compose.yml` | Local development services (PostgreSQL) |

## Local development

Prerequisites: Docker, Node.js, [uv](https://docs.astral.sh/uv/).

```bash
cp .env.example .env            # then set POSTGRES_PASSWORD
docker compose up -d            # PostgreSQL on 127.0.0.1:5432

cd api && uv run uvicorn forglemmigej_api.app:app --reload   # API on :8000
cd web && npm run dev                                       # web app on :5173, proxies /api to :8000
```

## Learning content

Downloaded data lives in `data/raw/` and is never committed:

```bash
curl -L -o data/raw/da_50k.txt https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/da/da_50k.txt
curl -L -o data/raw/kaikki-danish.jsonl https://kaikki.org/dictionary/Danish/kaikki.org-dictionary-Danish.jsonl
cd api && uv run python -m forglemmigej_api.content.importer --words 500   # --dry-run to preview
```

## Attribution

- Word frequencies: [FrequencyWords](https://github.com/hermitdave/FrequencyWords) by Hermit Dave, built from [OpenSubtitles 2018](http://opus.nlpl.eu/OpenSubtitles2018.php), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
- Meanings, gender, pronunciation and phrases: [Wiktionary](https://en.wiktionary.org/), via the [kaikki.org](https://kaikki.org/) extraction ([Wiktextract](https://github.com/tatuylonen/wiktextract)), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
