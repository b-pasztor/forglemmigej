from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

mlflow = pytest.importorskip("mlflow")  # the ml dependency group
pd = pytest.importorskip("pandas")

from forglemmigej_api.models import (
    AnswerMode,
    Card,
    CardType,
    Item,
    ItemKind,
    MemoryModel,
    ModelRole,
    Prediction,
    Review,
)
from forglemmigej_api.training.nightly import (
    EXPERIMENT,
    evaluate_models,
    run,
    shadow_predictions,
)

NOW = datetime(2026, 10, 20, 17, 30, tzinfo=UTC)


def add_reviews(db: Session, plan: list[tuple[float, int, dict[str, float]]]) -> None:
    """plan: (days before NOW, grade, {model_version: predicted recall})."""
    item = Item(
        kind=ItemKind.WORD, danish="nat", english="night", source="t", source_ref="nat"
    )
    db.add(item)
    db.flush()
    card = Card(item_id=item.id, card_type=CardType.RECOGNITION)
    db.add(card)
    db.flush()
    for days_ago, grade, predictions in plan:
        review = Review(
            card_id=card.id,
            reviewed_at=NOW - timedelta(days=days_ago),
            grade=grade,
            answer_mode=AnswerMode.SELF_GRADE,
        )
        db.add(review)
        db.flush()
        for version, value in predictions.items():
            db.add(
                Prediction(
                    review_id=review.id,
                    model_version=version,
                    predicted_recall=value,
                    is_champion=version == "fsrs6-default",
                )
            )
    db.flush()


PLAN = [
    (10, 3, {}),  # first exposure: no predictions
    (9.99, 3, {"fsrs6-default": 0.99, "fitted": 0.98}),  # minutes later: excluded
    (7, 1, {"fsrs6-default": 0.9, "fitted": 0.6}),
    (4, 3, {"fsrs6-default": 0.8, "fitted": 0.85}),
    (1, 3, {"fsrs6-default": 0.85}),  # only the champion predicted: excluded
]


def test_shadow_predictions_carry_the_gap_since_the_previous_review(
    db: Session,
) -> None:
    add_reviews(db, PLAN)
    frame = shadow_predictions(db, NOW - timedelta(days=14))
    gaps = sorted(frame.drop_duplicates("review_id")["gap_days"].round(2))
    assert gaps == [0.01, 2.99, 3.0, 3.0]


def test_models_are_compared_on_the_same_reviews_a_day_apart(db: Session) -> None:
    add_reviews(db, PLAN)
    scores = evaluate_models(shadow_predictions(db, NOW - timedelta(days=14)))
    assert set(scores.index) == {"fsrs6-default", "fitted"}
    assert scores.loc["fsrs6-default", "n"] == scores.loc["fitted", "n"] == 2
    # The fitted model gave the forgotten card a lower probability: better log loss.
    assert scores.loc["fitted", "log_loss"] < scores.loc["fsrs6-default", "log_loss"]


def test_nightly_run_logs_to_mlflow_and_skips_fitting_without_data(
    db: Session, tmp_path
) -> None:
    add_reviews(db, PLAN)
    db.add(
        MemoryModel(
            version="fitted", kind="fsrs", role=ModelRole.CHALLENGER, parameters=[0.1]
        )
    )
    db.flush()
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run() as active:
        result = run(db, NOW, min_reviews=1000)
    logged = mlflow.get_run(active.info.run_id).data

    assert result["fitted"] is None
    assert logged.tags["fit"].startswith("skipped:")
    assert logged.params["champion"] == "fsrs6-default"
    assert logged.metrics["data.reviews"] >= len(PLAN)
    assert logged.metrics["eval.champion.n"] == 2
    assert logged.tags["eval.challenger.fsrs.version"] == "fitted"


def test_nightly_runs_the_gate_and_keeps_a_young_fitted_challenger(
    db: Session, tmp_path
) -> None:
    add_reviews(db, PLAN)
    db.add(
        MemoryModel(
            version="fsrs-fitted-young",
            kind="fsrs",
            role=ModelRole.CHALLENGER,
            parameters=[0.1],
            created_at=NOW - timedelta(days=3),
        )
    )
    db.flush()
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run() as active:
        result = run(db, NOW, min_reviews=0)
    logged = mlflow.get_run(active.info.run_id).data

    # Too few shared reviews: the champion stays, nothing is refitted yet.
    assert result["gate"] == {"fsrs-fitted-young": False}
    assert result["promoted"] is None
    assert result["fitted"] is None
    assert logged.tags["gate.fsrs.decision"].startswith("keep:")
    assert logged.tags["fit"].startswith("skipped: the fitted challenger")


def test_models_outside_the_active_set_do_not_shrink_the_comparison(
    db: Session,
) -> None:
    add_reviews(db, PLAN)
    frame = shadow_predictions(db, NOW - timedelta(days=14))
    counted = frame[frame["gap_days"] >= 1].iloc[[0]]
    retired = counted.assign(model_version="retired-model")
    frame = pd.concat([frame, retired])
    # Unfiltered, only the review the retired model also predicted is left.
    assert evaluate_models(frame).loc["fsrs6-default", "n"] == 1
    scores = evaluate_models(frame, ["fsrs6-default", "fitted"])
    assert scores.loc["fsrs6-default", "n"] == 2


def test_shadow_predictions_of_an_empty_log(db: Session) -> None:
    frame = shadow_predictions(db, NOW - timedelta(days=14))
    assert frame.empty and "gap_days" in frame
