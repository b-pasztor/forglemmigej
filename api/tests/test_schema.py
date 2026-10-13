from datetime import UTC, datetime

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session

from forglemmigej_api.models import (
    AnswerMode,
    Card,
    CardType,
    Grade,
    Item,
    ItemKind,
    Prediction,
    Review,
)


@pytest.fixture
def session(migrated_engine: Engine):
    with Session(migrated_engine) as session:
        yield session
        session.rollback()


def add_review(session: Session, grade: int = Grade.GOOD) -> Review:
    item = Item(
        kind=ItemKind.WORD,
        danish="hus",
        english="house",
        article="et",
        source="test",
        source_ref=f"hus-{datetime.now(UTC).timestamp()}",
    )
    session.add(item)
    session.flush()
    card = Card(item_id=item.id, card_type=CardType.RECOGNITION)
    session.add(card)
    session.flush()
    review = Review(
        card_id=card.id,
        reviewed_at=datetime.now(UTC),
        grade=grade,
        answer_mode=AnswerMode.SELF_GRADE,
    )
    session.add(review)
    session.flush()
    return review


def test_reviews_cannot_be_updated(session: Session) -> None:
    review = add_review(session)
    with pytest.raises(DBAPIError, match="append-only"):
        session.execute(
            text("UPDATE reviews SET grade = 1 WHERE id = :id"), {"id": review.id}
        )


def test_reviews_cannot_be_deleted(session: Session) -> None:
    review = add_review(session)
    with pytest.raises(DBAPIError, match="append-only"):
        session.execute(text("DELETE FROM reviews WHERE id = :id"), {"id": review.id})


def test_grade_must_be_on_fsrs_scale(session: Session) -> None:
    with pytest.raises(IntegrityError, match="grade_range"):
        add_review(session, grade=5)


def test_only_one_champion_prediction_per_review(session: Session) -> None:
    review = add_review(session)
    session.add(
        Prediction(
            review_id=review.id,
            model_version="fsrs-6",
            predicted_recall=0.9,
            is_champion=True,
        )
    )
    session.add(
        Prediction(
            review_id=review.id,
            model_version="lgbm-1",
            predicted_recall=0.8,
            is_champion=False,
        )
    )
    session.flush()
    session.add(
        Prediction(
            review_id=review.id,
            model_version="lgbm-2",
            predicted_recall=0.7,
            is_champion=True,
        )
    )
    with pytest.raises(IntegrityError, match="one_champion"):
        session.flush()


def test_migrations_downgrade_and_upgrade_cleanly(
    migrated_engine: Engine, alembic_cfg: Config
) -> None:
    migrated_engine.dispose()  # release pooled connections before dropping tables
    command.downgrade(alembic_cfg, "base")
    command.upgrade(alembic_cfg, "head")
