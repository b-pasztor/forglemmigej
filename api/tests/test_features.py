import math
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from forglemmigej_api.features import (
    CARD_FEATURES,
    HISTORY_FEATURES,
    CardInfo,
    FutureLeakError,
    PastReview,
    card_features,
)
from forglemmigej_api.models import (
    AnswerMode,
    Card,
    CardType,
    Item,
    ItemKind,
    Review,
)
from forglemmigej_api.training.dataset import (
    examples_from_database,
    examples_from_history,
    load_card_history,
    load_card_info,
)

T0 = datetime(2026, 10, 1, 8, 0, tzinfo=UTC)


def review(days: float, grade: int, ms: int | None = 2000) -> PastReview:
    return PastReview(T0 + timedelta(days=days), grade, ms)


def test_history_features() -> None:
    history = [review(0, 3), review(1, 1), review(2, 3), review(5, 3, ms=None)]
    f = card_features(history, T0 + timedelta(days=8))
    assert set(f) == set(HISTORY_FEATURES)
    assert f["elapsed_days"] == pytest.approx(3)
    assert f["n_reviews"] == 4
    assert f["n_lapses"] == 1
    assert f["last_recalled"] == 1
    assert f["recall_rate"] == pytest.approx(0.75)
    assert f["success_streak"] == 2
    assert f["prev_interval_days"] == pytest.approx(3)
    assert f["days_since_first"] == pytest.approx(8)
    assert math.isnan(f["log_last_response_s"])
    assert 0 < f["fsrs_recall"] < 1


def test_fsrs_recall_drops_with_time() -> None:
    history = [review(0, 3), review(1, 3)]
    soon = card_features(history, T0 + timedelta(days=2))["fsrs_recall"]
    later = card_features(history, T0 + timedelta(days=30))["fsrs_recall"]
    assert soon > later


def test_future_reviews_are_refused() -> None:
    with pytest.raises(FutureLeakError):
        card_features([review(0, 3), review(5, 3)], T0 + timedelta(days=5))


def test_card_features() -> None:
    card = CardInfo(
        card_type="cloze", has_context=True, word_length=3, frequency_rank=345
    )
    f = card_features([review(0, 3)], T0 + timedelta(days=1, hours=4), card)
    assert set(f) == set(HISTORY_FEATURES) | set(CARD_FEATURES)
    assert (f["is_cloze"], f["is_recall"], f["has_context"], f["word_length"]) == (
        1,
        0,
        1,
        3,
    )
    assert f["log_frequency_rank"] == pytest.approx(math.log(345))
    assert f["hour_of_day"] == 14  # 12:00 UTC is 14:00 in Zurich (summer time)


def test_examples_skip_the_first_exposure() -> None:
    reviews = [review(0, 3), review(1, 1), review(3, 3)]
    examples = list(examples_from_history("c1", reviews))
    assert [(e.reviewed_at, e.recalled) for e in examples] == [
        (reviews[1].reviewed_at, False),
        (reviews[2].reviewed_at, True),
    ]
    assert examples[0].features["n_reviews"] == 1


def test_training_and_serving_paths_produce_identical_features(db: Session) -> None:
    """The train/serve skew check: building the dataset from the whole log must
    give the same features as computing them one review at a time, the way the
    scheduler will at answer time."""
    item = Item(
        kind=ItemKind.WORD,
        danish="hus",
        english="house",
        frequency_rank=345,
        source="test",
        source_ref="hus|parity",
    )
    db.add(item)
    db.flush()
    card = Card(item_id=item.id, card_type=CardType.RECOGNITION)
    db.add(card)
    db.flush()
    for days, grade in [(0, 3), (0.01, 3), (2, 1), (2.01, 3), (6, 3)]:
        db.add(
            Review(
                card_id=card.id,
                reviewed_at=T0 + timedelta(days=days),
                grade=grade,
                response_time_ms=1500,
                answer_mode=AnswerMode.SELF_GRADE,
            )
        )
    db.flush()

    training = [e for e in examples_from_database(db) if e.card_id == str(card.id)]
    assert len(training) == 4
    info = load_card_info(db, card.id)
    for example in training:
        serving = card_features(
            load_card_history(db, card.id, example.reviewed_at),
            example.reviewed_at,
            info,
        )
        assert serving == example.features
