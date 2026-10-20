"""Memory models: the FSRS champion that schedules, and challengers in shadow mode.

FSRS keeps two numbers per card (stability and difficulty) and from them
predicts the probability of recall at any moment ("retrievability"). A card is
due when that probability would drop below the desired retention (90%). About
20 parameters decide how those numbers change after each answer; a challenger
can be FSRS with parameters fitted to the learner's own reviews.

Which models are active comes from the `memory_models` registry. Fuzzing
(randomly spreading due dates) is off, so scheduling is deterministic: card
state can be rebuilt by replaying the review log, and tests are exact.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

from fsrs import Card as FsrsCard
from fsrs import Rating, Scheduler, State
from sqlalchemy import select
from sqlalchemy.orm import Session

from forglemmigej_api.features import PastReview
from forglemmigej_api.models import CardStatus, Grade, MemoryModel, ModelRole

_STATUS = {
    State.Learning: CardStatus.LEARNING,
    State.Review: CardStatus.REVIEW,
    State.Relearning: CardStatus.LEARNING,
}


@dataclass(frozen=True)
class FsrsModel:
    version: str
    parameters: tuple[float, ...]

    @property
    def scheduler(self) -> Scheduler:
        return _scheduler(self.parameters)


@lru_cache(maxsize=16)
def _scheduler(parameters: tuple[float, ...]) -> Scheduler:
    return Scheduler(parameters=parameters, enable_fuzzing=False)


class NoChampionError(RuntimeError):
    pass


def active_models(session: Session) -> tuple[FsrsModel, list[FsrsModel]]:
    """The champion and the challengers, from the model registry."""
    rows = session.scalars(
        select(MemoryModel)
        .where(MemoryModel.role.in_([ModelRole.CHAMPION, ModelRole.CHALLENGER]))
        .order_by(MemoryModel.id)
    ).all()
    models = {
        row.version: (row.role, FsrsModel(row.version, tuple(row.parameters)))
        for row in rows
    }
    champions = [m for role, m in models.values() if role == ModelRole.CHAMPION]
    if not champions:
        raise NoChampionError("the model registry has no champion")
    return champions[0], [
        m for role, m in models.values() if role == ModelRole.CHALLENGER
    ]


def predict_from_state(
    model: FsrsModel, state: dict[str, Any] | None, at: datetime
) -> float | None:
    """The champion's prediction from the card's cached state, or None before the
    first review (the first exposure is learning, not a recall test)."""
    if not state:
        return None
    card = FsrsCard.from_dict(state)
    if card.last_review is None:
        return None
    return float(model.scheduler.get_card_retrievability(card, at))


@lru_cache(maxsize=100_000)
def _replay(
    parameters: tuple[float, ...], reviews: tuple[tuple[datetime, int], ...]
) -> FsrsCard:
    """FSRS state after replaying `reviews` with the given parameters. Cached per
    prefix, so a growing history costs one step per new review."""
    card = _replay(parameters, reviews[:-1]) if len(reviews) > 1 else FsrsCard()
    reviewed_at, grade = reviews[-1]
    card, _ = _scheduler(parameters).review_card(card, Rating(grade), reviewed_at)
    return card


def predict_from_history(
    model: FsrsModel, history: Sequence[PastReview], at: datetime
) -> float | None:
    """A model's prediction by replaying the card's review history (shadow mode:
    challengers have no cached state of their own). Uses only reviews before `at`."""
    past = tuple(
        (r.reviewed_at.astimezone(UTC), r.grade) for r in history if r.reviewed_at < at
    )
    if not past:
        return None
    card = _replay(model.parameters, past)
    return float(model.scheduler.get_card_retrievability(card, at.astimezone(UTC)))


def apply_review(
    model: FsrsModel, state: dict[str, Any] | None, grade: Grade, at: datetime
) -> tuple[dict[str, Any], datetime, CardStatus]:
    """New FSRS state, due time and status after a review (champion only)."""
    card = FsrsCard.from_dict(state) if state else FsrsCard()
    card, _log = model.scheduler.review_card(card, Rating(int(grade)), at)
    return card.to_dict(), card.due, _STATUS[card.state]
