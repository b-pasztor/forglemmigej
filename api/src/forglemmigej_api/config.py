from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# The repository root holds the shared .env used by docker compose.
_REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_REPO_ROOT / ".env", extra="ignore")

    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    postgres_db: str = "forglemmigej"
    postgres_user: str = "forglemmigej"
    postgres_password: str

    # Learning settings.
    timezone: str = "Europe/Zurich"  # where "today" starts and ends
    # New words per day; a word's second card (cloze/recall) does not count.
    new_words_per_day: int = 15

    def database_url(self, database: str | None = None) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{database or self.postgres_db}"
        )


def get_settings() -> Settings:
    return Settings()  # password comes from the environment
