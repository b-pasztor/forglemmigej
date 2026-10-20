"""Pre-train a LightGBM recall model on the public review logs and register it as
a shadow challenger.

Usage (from api/; needs data/processed/public_examples.parquet, see build_public):
    uv run --group ml python -m forglemmigej_api.training.pretrain_lightgbm
    uv run --group ml python -m forglemmigej_api.training.pretrain_lightgbm --register

First an honest score: train on the earlier 80% of each user's reviews, test on
the later 20%, next to default FSRS on the same reviews. Then the registered
model is refitted on all the data. It uses the history features only: the
public logs have no response times and no card details.

With MLFLOW_TRACKING_URI set, the run (scores and model) is logged to MLflow.
"""

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from sqlalchemy import update
from sqlalchemy.orm import Session

from forglemmigej_api.features import HISTORY_FEATURES
from forglemmigej_api.models import MemoryModel, ModelRole
from forglemmigej_api.training.evaluate import chronological_split, score
from forglemmigej_api.training.models import fit_lightgbm, predict

REPO_ROOT = Path(__file__).resolve().parents[4]
PREFIX = "lightgbm-public-"
# Always NaN in the public logs, so the model could not learn anything from it.
FEATURES = tuple(f for f in HISTORY_FEATURES if f != "log_last_response_s")


def holdout_scores(data: pd.DataFrame) -> pd.DataFrame:
    """LightGBM vs default FSRS on the later 20% of every user's reviews."""
    train, test = chronological_split(data, test_fraction=0.2, group="user")
    model = fit_lightgbm(train, FEATURES)
    return pd.DataFrame(
        {
            "lightgbm": score(
                test["recalled"], predict(model, test, FEATURES)
            ).as_dict(),
            "fsrs-default": score(test["recalled"], test["fsrs_recall"]).as_dict(),
        }
    ).T


def register_challenger(
    session: Session, model_text: str, data: pd.DataFrame, version: str
) -> MemoryModel:
    """Add the model as a challenger; earlier pre-trained LightGBM challengers retire."""
    session.execute(
        update(MemoryModel)
        .where(
            MemoryModel.role == ModelRole.CHALLENGER,
            MemoryModel.version.startswith(PREFIX),
        )
        .values(role=ModelRole.RETIRED)
    )
    model = MemoryModel(
        version=version,
        kind="lightgbm",
        role=ModelRole.CHALLENGER,
        parameters={"features": list(FEATURES), "model": model_text},
        trained_on_reviews=len(data),
        trained_until=data["reviewed_at"].max().to_pydatetime(),
        notes=(
            "LightGBM pre-trained on public Anki review logs "
            f"(open-spaced-repetition/fsrs-dataset, {data['user'].nunique()} users); "
            "history features only. Shadow mode: predicts, never schedules."
        ),
    )
    session.add(model)
    session.flush()
    return model


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data/processed/public_examples.parquet",
    )
    parser.add_argument("--version", default=f"{PREFIX}v1")
    parser.add_argument(
        "--register", action="store_true", help="register it in the database"
    )
    args = parser.parse_args(argv)

    data = pd.read_parquet(args.data)
    print(f"{len(data):,} examples from {data['user'].nunique()} users")
    scores = holdout_scores(data)
    print(scores.round(4).to_string())

    model = fit_lightgbm(data, FEATURES)
    model_text = model.booster_.model_to_string()
    print(
        f"final model: {model.booster_.num_trees()} trees, {len(model_text) / 1e6:.1f} MB"
    )

    if os.environ.get("MLFLOW_TRACKING_URI"):
        import mlflow

        from forglemmigej_api.training.nightly import EXPERIMENT

        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=f"pretrain-{args.version}"):
            mlflow.log_params(
                {
                    "version": args.version,
                    "examples": len(data),
                    "features": ",".join(FEATURES),
                }
            )
            for name, row in scores.iterrows():
                mlflow.log_metrics(
                    {f"holdout.{name}.{k}": float(v) for k, v in row.items()}
                )
            mlflow.log_text(model_text, "model.txt")

    if args.register:
        from forglemmigej_api.db import SessionLocal

        with SessionLocal().begin() as session:
            registered = register_challenger(session, model_text, data, args.version)
            print(f"registered challenger {registered.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
