from datetime import UTC, datetime, timedelta

import lightgbm
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from forglemmigej_api.features import PastReview, card_features
from forglemmigej_api.models import (
    Card,
    CardType,
    Grade,
    MemoryModel,
    ModelRole,
    Prediction,
)
from forglemmigej_api.scheduling import (
    NoChampionError,
    active_models,
    apply_review,
    predict_from_history,
    predict_from_state,
)
from forglemmigej_api.training.dataset import load_card_info

from .test_review_api import Clock, add_word, client, clock  # noqa: F401 - fixtures

T0 = datetime(2026, 10, 1, 8, 0, tzinfo=UTC)


def add_challenger(db: Session, version: str, scale: float = 1.0) -> MemoryModel:
    champion = db.scalars(
        select(MemoryModel).where(MemoryModel.role == ModelRole.CHAMPION)
    ).one()
    params = list(champion.parameters)
    # w0..w3 are the initial stabilities after Again/Hard/Good/Easy; the tests
    # answer Good, so change w2 to get different predictions.
    params[2] *= scale
    model = MemoryModel(
        version=version, kind="fsrs", role=ModelRole.CHALLENGER, parameters=params
    )
    db.add(model)
    db.flush()
    return model


def test_registry_starts_with_default_fsrs_as_champion(db: Session) -> None:
    champion, challengers = active_models(db)
    assert champion.version == "fsrs6-default"
    assert len(champion.parameters) == 21
    assert challengers == []


def test_only_one_champion(db: Session) -> None:
    db.add(
        MemoryModel(version="x", kind="fsrs", role=ModelRole.CHAMPION, parameters=[0.1])
    )
    with pytest.raises(IntegrityError):
        db.flush()


def test_cached_state_and_replayed_history_agree(db: Session) -> None:
    """The champion predicts from the card's cached state, challengers by
    replaying the log. With the same parameters both must give the same number,
    which is also what makes the cache rebuildable from the log."""
    champion, _ = active_models(db)
    state, history = None, []
    for days, grade in [(0, 3), (0.01, 3), (2, 1), (2.01, 3)]:
        at = T0 + timedelta(days=days)
        state, _, _ = apply_review(champion, state, Grade(grade), at)
        history.append(PastReview(at, grade))
    later = T0 + timedelta(days=9)
    assert predict_from_state(champion, state, later) == pytest.approx(
        predict_from_history(champion, history, later)
    )


def test_challengers_predict_in_shadow_mode(
    client: TestClient,  # noqa: F811
    db: Session,
    clock: Clock,  # noqa: F811
) -> None:
    add_challenger(db, "fsrs-test-challenger", scale=3.0)
    ids = add_word(db, "hus", 1)
    url = f"/api/cards/{ids[CardType.RECOGNITION]}/answer"
    client.post(url, json={"grade": 3})  # first exposure: nobody predicts
    # Minutes after learning, every model says ~100%; days later they differ.
    clock.advance(days=3)
    client.post(url, json={"grade": 3})

    rows = db.scalars(select(Prediction).order_by(Prediction.is_champion)).all()
    assert [(p.model_version, p.is_champion) for p in rows] == [
        ("fsrs-test-challenger", False),
        ("fsrs6-default", True),
    ]
    assert rows[0].review_id == rows[1].review_id
    assert rows[0].predicted_recall != pytest.approx(rows[1].predicted_recall)

    # Scheduling follows the champion only.
    champion, _ = active_models(db)
    card = db.get(Card, ids[CardType.RECOGNITION])
    assert predict_from_state(champion, card.scheduler_state, clock.now) is not None


def test_retired_models_stop_predicting(db: Session) -> None:
    model = add_challenger(db, "fsrs-old")
    model.role = ModelRole.RETIRED
    db.flush()
    _, challengers = active_models(db)
    assert challengers == []


def tiny_lightgbm(features: tuple[str, ...]) -> lightgbm.Booster:
    """A small booster on synthetic data: recalled when reviewed again soon."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=(500, len(features)))
    x[:, 0] = rng.uniform(0, 30, size=500)  # elapsed_days
    y = (x[:, 0] < 5).astype(int)
    data = lightgbm.Dataset(x, y, feature_name=list(features))
    params = {"objective": "binary", "verbose": -1, "num_threads": 1}
    return lightgbm.train(params, data, num_boost_round=10)


def test_lightgbm_challenger_predicts_from_the_shared_features(
    client: TestClient,  # noqa: F811
    db: Session,
    clock: Clock,  # noqa: F811
) -> None:
    features = ("elapsed_days", "n_reviews", "fsrs_recall", "word_length")
    booster = tiny_lightgbm(features)
    db.add(
        MemoryModel(
            version="lightgbm-test",
            kind="lightgbm",
            role=ModelRole.CHALLENGER,
            parameters={"features": list(features), "model": booster.model_to_string()},
        )
    )
    db.flush()
    ids = add_word(db, "hus", 1)
    url = f"/api/cards/{ids[CardType.RECOGNITION]}/answer"
    first = clock.now
    client.post(url, json={"grade": 3})
    clock.advance(days=3)
    client.post(url, json={"grade": 3})

    shadow = db.scalars(
        select(Prediction).where(Prediction.model_version == "lightgbm-test")
    ).one()
    # The same number as the shared feature function + the booster (no skew).
    expected = card_features(
        [PastReview(first, 3)], clock.now, load_card_info(db, ids[CardType.RECOGNITION])
    )
    row = np.array([[expected[name] for name in features]])
    assert shadow.predicted_recall == pytest.approx(booster.predict(row)[0])


def test_only_fsrs_can_be_champion(db: Session) -> None:
    champion = db.scalars(
        select(MemoryModel).where(MemoryModel.role == ModelRole.CHAMPION)
    ).one()
    champion.role = ModelRole.RETIRED
    db.flush()
    booster = tiny_lightgbm(("elapsed_days",))
    db.add(
        MemoryModel(
            version="lightgbm-champion",
            kind="lightgbm",
            role=ModelRole.CHAMPION,
            parameters={
                "features": ["elapsed_days"],
                "model": booster.model_to_string(),
            },
        )
    )
    db.flush()
    with pytest.raises(NoChampionError):
        active_models(db)
