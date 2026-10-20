from datetime import timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

pd = pytest.importorskip("pandas")  # the ml dependency group

from forglemmigej_api.features import PastReview
from forglemmigej_api.models import (
    Card,
    CardType,
    MemoryModel,
    ModelPromotion,
    ModelRole,
    Review,
)
from forglemmigej_api.scheduling import (
    active_models,
    predict_from_history,
    predict_from_state,
)
from forglemmigej_api.training.gate import (
    MIN_REVIEWS,
    bootstrap_p_better,
    decide,
    promote,
)

from .test_review_api import Clock, add_word, client, clock  # noqa: F401 - fixtures
from .test_shadow import add_challenger, tiny_lightgbm


def shadow_frame(
    champion: float, challenger: float, n_cards: int = 200, recall: float = 0.85
) -> pd.DataFrame:
    """Two reviews per card, a day or more apart; each model predicts a constant."""
    rng = np.random.default_rng(1)
    rows = []
    for review_id in range(n_cards * 2):
        recalled = int(rng.random() < recall)
        for version, p in [("champ", champion), ("chal", challenger)]:
            rows.append(
                {
                    "review_id": review_id,
                    "card_id": review_id // 2,
                    "model_version": version,
                    "predicted_recall": p,
                    "recalled": recalled,
                    "gap_days": 2.0,
                }
            )
    return pd.DataFrame(rows)


def test_a_clearly_better_fsrs_challenger_is_promoted() -> None:
    decision = decide(shadow_frame(0.60, 0.85), "champ", "chal", "fsrs")
    assert decision.promote, decision.reasons
    assert decision.n == 400
    assert decision.p_better > 0.99
    assert decision.challenger_log_loss < decision.champion_log_loss


def test_lightgbm_is_compared_but_never_promoted() -> None:
    decision = decide(shadow_frame(0.60, 0.85), "champ", "chal", "lightgbm")
    assert not decision.promote
    assert decision.reasons == ["lightgbm models cannot schedule"]
    assert decision.p_better > 0.99  # the comparison still happens


def test_too_few_reviews_keep_the_champion() -> None:
    decision = decide(shadow_frame(0.60, 0.85, n_cards=50), "champ", "chal", "fsrs")
    assert not decision.promote
    assert decision.reasons == [f"100 shared reviews < {MIN_REVIEWS}"]


def test_reviews_minutes_after_learning_do_not_count() -> None:
    frame = shadow_frame(0.60, 0.85)
    frame["gap_days"] = 0.01
    decision = decide(frame, "champ", "chal", "fsrs")
    assert decision.n == 0
    assert not decision.promote


def test_a_tie_is_not_good_enough() -> None:
    decision = decide(shadow_frame(0.85, 0.85), "champ", "chal", "fsrs")
    assert not decision.promote
    assert decision.p_better == 0.0


def test_better_log_loss_but_worse_calibration_is_rejected() -> None:
    """Half the cards are easy (95% recall), half hard (75%). The champion says
    85% for all: no ranking, but calibrated on average. The challenger tells easy
    from hard (lower log loss) but is overconfident (97% / 84%), and around the
    90% threshold that would schedule hard cards too late."""
    rng = np.random.default_rng(2)
    rows = []
    for review_id in range(2000):
        easy = review_id % 2 == 0
        recalled = int(rng.random() < (0.95 if easy else 0.75))
        for version, p in [("champ", 0.85), ("chal", 0.97 if easy else 0.84)]:
            rows.append(
                {
                    "review_id": review_id,
                    "card_id": review_id // 2,
                    "model_version": version,
                    "predicted_recall": p,
                    "recalled": recalled,
                    "gap_days": 2.0,
                }
            )
    decision = decide(pd.DataFrame(rows), "champ", "chal", "fsrs")
    assert decision.challenger_log_loss < decision.champion_log_loss
    assert not decision.promote
    assert [r for r in decision.reasons if "worse calibrated" in r]


def test_bootstrap_resamples_whole_cards() -> None:
    # One card carries the whole difference: resampling cards makes the result
    # depend on whether that card is drawn, so confidence is far from certain.
    cards = np.array([0] * 50 + list(range(1, 51)))
    champion = np.array([1.0] * 50 + [0.5] * 50)
    challenger = np.array([0.1] * 50 + [0.5] * 50)
    assert 0.5 < bootstrap_p_better(cards, champion, challenger) < 0.8


def test_promotion_rebuilds_card_state_with_the_new_champion(
    client: TestClient,  # noqa: F811
    db: Session,
    clock: Clock,  # noqa: F811
) -> None:
    challenger = add_challenger(db, "fsrs-better", scale=3.0)
    ids = add_word(db, "hus", 1)
    card_id = ids[CardType.RECOGNITION]
    url = f"/api/cards/{card_id}/answer"
    for days, grade in [(0, 3), (2, 1), (1, 3)]:
        clock.advance(days=days)
        client.post(url, json={"grade": grade})

    promotion = promote(db, challenger.version, "manual: test")

    assert (promotion.previous_version, promotion.version) == (
        "fsrs6-default",
        "fsrs-better",
    )
    assert promotion.cards_rebuilt == 1
    champion, challengers = active_models(db)
    assert champion.version == "fsrs-better"
    assert challengers == []  # the old champion retired
    # The cache is now what the new champion computes from the log.
    card = db.get(Card, card_id)
    history = [
        PastReview(r.reviewed_at, r.grade)
        for r in db.scalars(select(Review).where(Review.card_id == card_id))
    ]
    later = clock.now + timedelta(days=5)
    assert predict_from_state(champion, card.scheduler_state, later) == pytest.approx(
        predict_from_history(champion, history, later)
    )
    assert db.scalars(select(ModelPromotion)).one().reason == "manual: test"


def test_only_fsrs_can_be_promoted(db: Session) -> None:
    db.add(
        MemoryModel(
            version="lgbm",
            kind="lightgbm",
            role=ModelRole.CHALLENGER,
            parameters={
                "features": ["elapsed_days"],
                "model": tiny_lightgbm(("elapsed_days",)).model_to_string(),
            },
        )
    )
    db.flush()
    with pytest.raises(ValueError, match="cannot schedule"):
        promote(db, "lgbm", "manual: test")
