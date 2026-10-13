from collections.abc import Iterator
from functools import cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from forglemmigej_api.config import get_settings


@cache
def get_engine() -> Engine:
    return create_engine(
        get_settings().database_url(), connect_args={"connect_timeout": 5}
    )


def SessionLocal() -> sessionmaker[Session]:
    return sessionmaker(get_engine())


def get_session() -> Iterator[Session]:
    """FastAPI dependency: one transaction per request."""
    with SessionLocal().begin() as session:
        yield session
