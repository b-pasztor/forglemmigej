from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

pytest.importorskip("torch")  # the FSRS optimizer comes with the ml dependency group

from forglemmigej_api.features import PastReview
from forglemmigej_api.models import (
    AnswerMode,
    Card,
    CardType,
    Item,
    ItemKind,
    MemoryModel,
    ModelRole,
    Review,
)
from forglemmigej_api.scheduling import active_models
from forglemmigej_api.training.fit_fsrs import (
    fit_reviews,
    histories_from_database,
    register_challenger,
    review_logs,
)

T0 = datetime(2026, 10, 1, 8, 0, tzinfo=UTC)


def test_histories_respect_the_cutoff(db: Session) -> None:
    item = Item(
        kind=ItemKind.WORD, danish="fit", english="fit", source="t", source_ref="fit"
    )
    db.add(item)
    db.flush()
    card = Card(item_id=item.id, card_type=CardType.RECOGNITION)
    db.add(card)
    db.flush()
    for days in (0, 1, 5):
        db.add(
            Review(
                card_id=card.id,
                reviewed_at=T0 + timedelta(days=days),
                grade=3,
                answer_mode=AnswerMode.SELF_GRADE,
            )
        )
    db.flush()

    histories = histories_from_database(db, until=T0 + timedelta(days=2))
    assert [r.reviewed_at for r in histories[card.id]] == [T0, T0 + timedelta(days=1)]
    logs = review_logs(histories)
    assert [(log.card_id, int(log.rating)) for log in logs] == [
        (card.id, 3),
        (card.id, 3),
    ]


def test_a_new_fit_retires_the_previous_fitted_challenger(db: Session) -> None:
    champion, _ = active_models(db)
    first = register_challenger(db, champion.parameters, 1200, T0)
    second = register_challenger(db, champion.parameters, 1500, T0 + timedelta(days=7))

    roles = dict(
        db.execute(
            select(MemoryModel.version, MemoryModel.role).where(
                MemoryModel.version.in_([first.version, second.version])
            )
        ).all()
    )
    assert roles == {
        first.version: ModelRole.RETIRED,
        second.version: ModelRole.CHALLENGER,
    }
    assert (second.trained_on_reviews, second.trained_until) == (
        1500,
        T0 + timedelta(days=7),
    )
    _, challengers = active_models(db)
    assert [c.version for c in challengers] == [second.version]


def test_only_reviews_a_day_apart_are_usable_for_fitting() -> None:
    history = [
        PastReview(T0 + timedelta(days=days), 3)
        for days in (0, 0.01, 0.5, 1.6, 2.0, 5.0)
    ]
    # 0.01 and 0.5 are same-day; 1.6 is a day+ later; 2.0 is not; 5.0 is.
    assert fit_reviews({1: history}) == 2
