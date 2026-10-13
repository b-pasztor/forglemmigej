from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text

from forglemmigej_api.config import get_settings

TEST_DATABASE = "forglemmigej_test"
API_ROOT = Path(__file__).resolve().parents[1]


def alembic_config(url: str) -> Config:
    config = Config(API_ROOT / "alembic.ini")
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    config.attributes["configure_logger"] = False
    return config


@pytest.fixture(scope="session")
def test_database_url() -> Iterator[str]:
    """A fresh database for the test run, dropped afterwards."""
    settings = get_settings()
    admin = create_engine(settings.database_url(), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f"DROP DATABASE IF EXISTS {TEST_DATABASE} WITH (FORCE)"))
        conn.execute(text(f"CREATE DATABASE {TEST_DATABASE}"))
    yield settings.database_url(TEST_DATABASE)
    with admin.connect() as conn:
        conn.execute(text(f"DROP DATABASE IF EXISTS {TEST_DATABASE} WITH (FORCE)"))
    admin.dispose()


@pytest.fixture(scope="session")
def alembic_cfg(test_database_url: str) -> Config:
    return alembic_config(test_database_url)


@pytest.fixture(scope="session")
def migrated_engine(test_database_url: str, alembic_cfg: Config) -> Iterator[Engine]:
    command.upgrade(alembic_cfg, "head")
    engine = create_engine(test_database_url)
    yield engine
    engine.dispose()
