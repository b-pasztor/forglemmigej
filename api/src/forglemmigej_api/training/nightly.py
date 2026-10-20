"""The nightly training job: evaluate the models on recent reviews, fit a new
challenger when there is enough data, and record everything in MLflow.

Runs as a Kubernetes CronJob before the node's evening shutdown. Usage:
    MLFLOW_TRACKING_URI=http://mlflow:5000 \\
        uv run --group ml python -m forglemmigej_api.training.nightly

Shadow evaluation: every logged prediction is compared with what actually
happened. Only reviews at least a day after the card's previous review count;
minutes after learning every model predicts ~100%, which says nothing.
"""

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import mlflow
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from forglemmigej_api.models import Grade, MemoryModel, ModelRole, Prediction, Review
from forglemmigej_api.training.evaluate import score
from forglemmigej_api.training.fit_fsrs import (
    DEFAULT_MIN_REVIEWS,
    fit_parameters,
    histories_from_database,
    register_challenger,
)

EXPERIMENT = "forglemmigej-memory-model"
EVALUATION_DAYS = 14
MIN_GAP_DAYS = 1.0


def shadow_predictions(session: Session, since: datetime) -> pd.DataFrame:
    """One row per (review, model) since `since`: prediction, outcome, and the
    days since the card's previous review."""
    previous = (
        func.lag(Review.reviewed_at)
        .over(partition_by=Review.card_id, order_by=(Review.reviewed_at, Review.id))
        .label("previous_at")
    )
    reviews = select(Review.id, Review.reviewed_at, Review.grade, previous).subquery()
    rows = session.execute(
        select(
            reviews.c.id,
            reviews.c.reviewed_at,
            reviews.c.previous_at,
            reviews.c.grade,
            Prediction.model_version,
            Prediction.predicted_recall,
            Prediction.is_champion,
        )
        .join(Prediction, Prediction.review_id == reviews.c.id)
        .where(reviews.c.reviewed_at >= since)
    ).all()
    frame = pd.DataFrame(
        rows,
        columns=[
            "review_id", "reviewed_at", "previous_at", "grade",
            "model_version", "predicted_recall", "is_champion",
        ],
    )  # fmt: skip
    frame["recalled"] = (frame["grade"] > Grade.AGAIN).astype(int)
    frame["gap_days"] = (
        frame["reviewed_at"] - frame["previous_at"]
    ).dt.total_seconds() / 86_400
    return frame


def evaluate_models(predictions: pd.DataFrame) -> pd.DataFrame:
    """Scores per model, on the reviews that every model predicted (so the
    comparison is on exactly the same reviews)."""
    eligible = predictions[predictions["gap_days"] >= MIN_GAP_DAYS]
    if eligible.empty:
        return pd.DataFrame(columns=["log_loss", "brier", "auc", "rmse_bins", "n"])
    n_models = eligible["model_version"].nunique()
    shared = (
        eligible.groupby("review_id")["model_version"].transform("nunique") == n_models
    )
    eligible = eligible[shared]
    return pd.DataFrame(
        {
            version: score(part["recalled"], part["predicted_recall"]).as_dict()
            for version, part in eligible.groupby("model_version")
        }
    ).T


def data_summary(session: Session) -> dict[str, float]:
    total, recalled, cards, first, last = session.execute(
        select(
            func.count(),
            func.count().filter(Review.grade > Grade.AGAIN),
            func.count(Review.card_id.distinct()),
            func.min(Review.reviewed_at),
            func.max(Review.reviewed_at),
        )
    ).one()
    return {
        "reviews": total,
        "cards_reviewed": cards,
        "recall_rate": (recalled / total) if total else 0.0,
        "days_of_data": ((last - first).total_seconds() / 86_400) if total else 0.0,
    }


def run(session: Session, now: datetime, min_reviews: int) -> dict:
    """The nightly steps; returns a summary (also logged to MLflow)."""
    champion = session.scalars(
        select(MemoryModel.version).where(MemoryModel.role == ModelRole.CHAMPION)
    ).one()
    summary = data_summary(session)
    mlflow.log_params(
        {
            "data_cutoff": now.isoformat(),
            "min_reviews": min_reviews,
            "champion": champion,
        }
    )
    mlflow.log_metrics({f"data.{k}": float(v) for k, v in summary.items()})

    since = now - timedelta(days=EVALUATION_DAYS)
    scores = evaluate_models(shadow_predictions(session, since))
    for version, row in scores.iterrows():
        role = "champion" if version == champion else "challenger"
        mlflow.log_metrics(
            {f"eval.{role}.{metric}": float(row[metric]) for metric in scores.columns}
        )
        mlflow.set_tag(f"eval.{role}.version", version)
    mlflow.log_text(scores.to_csv(), "evaluation.csv")

    result = {
        "summary": summary,
        "evaluated_models": list(scores.index),
        "fitted": None,
    }
    histories = histories_from_database(session, until=now)
    n_reviews = sum(len(h) for h in histories.values())
    if n_reviews < min_reviews:
        mlflow.set_tag("fit", f"skipped: {n_reviews} reviews < {min_reviews}")
        return result

    parameters = fit_parameters(histories)
    model = register_challenger(session, parameters, n_reviews, now)
    mlflow.set_tag("fit", f"registered {model.version}")
    mlflow.log_dict(
        {"version": model.version, "parameters": parameters}, "challenger.json"
    )
    result["fitted"] = model.version
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-reviews", type=int, default=DEFAULT_MIN_REVIEWS)
    args = parser.parse_args(argv)

    from forglemmigej_api.db import SessionLocal

    now = datetime.now(UTC)
    mlflow.set_experiment(EXPERIMENT)
    with (
        mlflow.start_run(run_name=f"nightly-{now:%Y-%m-%d}"),
        SessionLocal().begin() as session,
    ):
        result = run(session, now, args.min_reviews)
    print(json.dumps(result, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
