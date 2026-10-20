"""Fit FSRS parameters to a learner's own reviews and register them as a challenger.

FSRS's ~20 default parameters were fitted on many Anki users. Fitting adjusts
them until FSRS's predictions match *this* learner's answers as closely as
possible (the optimizer minimises log loss with gradient descent, in PyTorch).

Usage (from api/):
    uv run --group ml python -m forglemmigej_api.training.fit_fsrs --dry-run
    uv run --group ml python -m forglemmigej_api.training.fit_fsrs
"""

import argparse
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from fsrs import Optimizer, Rating, ReviewLog
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from forglemmigej_api.features import PastReview
from forglemmigej_api.models import MemoryModel, ModelRole, Review

FITTED_PREFIX = "fsrs-fitted-"
# Below this the optimizer has too little to learn from and mostly returns
# the defaults; the FSRS authors suggest about a thousand reviews.
DEFAULT_MIN_REVIEWS = 1000


def review_logs(histories: Mapping[int, Sequence[PastReview]]) -> list[ReviewLog]:
    return [
        ReviewLog(
            card_id, Rating(r.grade), r.reviewed_at.astimezone(UTC), r.response_time_ms
        )
        for card_id, reviews in histories.items()
        for r in reviews
    ]


def fit_parameters(histories: Mapping[int, Sequence[PastReview]]) -> list[float]:
    return [
        float(p) for p in Optimizer(review_logs(histories)).compute_optimal_parameters()
    ]


def histories_from_database(
    session: Session, until: datetime | None = None
) -> dict[int, list[PastReview]]:
    """Card id -> chronological reviews, optionally only those before `until`."""
    query = select(Review).order_by(Review.card_id, Review.reviewed_at, Review.id)
    if until is not None:
        query = query.where(Review.reviewed_at < until)
    histories: dict[int, list[PastReview]] = {}
    for r in session.scalars(query):
        histories.setdefault(r.card_id, []).append(
            PastReview(r.reviewed_at, r.grade, r.response_time_ms)
        )
    return histories


def register_challenger(
    session: Session, parameters: Sequence[float], n_reviews: int, until: datetime
) -> MemoryModel:
    """Add the fitted model as a challenger; earlier fitted challengers retire, so
    one fitted model at a time is compared with the champion."""
    session.execute(
        update(MemoryModel)
        .where(
            MemoryModel.role == ModelRole.CHALLENGER,
            MemoryModel.version.startswith(FITTED_PREFIX),
        )
        .values(role=ModelRole.RETIRED)
    )
    model = MemoryModel(
        version=f"{FITTED_PREFIX}{until:%Y%m%d-%H%M}",
        kind="fsrs",
        role=ModelRole.CHALLENGER,
        parameters=list(parameters),
        trained_on_reviews=n_reviews,
        trained_until=until,
        notes="FSRS parameters fitted to the learner's own reviews.",
    )
    session.add(model)
    session.flush()
    return model


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-reviews", type=int, default=DEFAULT_MIN_REVIEWS)
    parser.add_argument(
        "--dry-run", action="store_true", help="fit and print, do not register"
    )
    args = parser.parse_args(argv)

    from forglemmigej_api.db import SessionLocal

    until = datetime.now(UTC)
    with SessionLocal().begin() as session:
        histories = histories_from_database(session, until)
        n_reviews = sum(len(h) for h in histories.values())
        print(
            f"{n_reviews} reviews of {len(histories)} cards before {until:%Y-%m-%d %H:%M} UTC"
        )
        if n_reviews < args.min_reviews:
            print(
                f"not enough reviews to fit (minimum {args.min_reviews}); nothing registered"
            )
            return 0
        parameters = fit_parameters(histories)
        print("fitted parameters:", [round(p, 4) for p in parameters])
        if args.dry_run:
            return 0
        model = register_challenger(session, parameters, n_reviews, until)
        print(f"registered challenger {model.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
