# forglemmigej API

FastAPI backend: review scheduling, review logging and model serving.

```bash
uv run uvicorn forglemmigej_api.app:app --reload   # dev server on :8000
uv run pytest                                      # tests
uv run ruff check .                                # lint
uv run alembic upgrade head                        # apply database migrations
uv run alembic revision --autogenerate -m "..."     # create a migration from model changes
```

Tests need the local PostgreSQL from `docker compose up -d`; they create and drop a separate `forglemmigej_test` database.
