"""Memory models: the FSRS champion that schedules, and challengers in shadow mode.

FSRS keeps two numbers per card (stability and difficulty) and from them
predicts the probability of recall at any moment ("retrievability"). A card is
due when that probability would drop below the desired retention (90%). About
20 parameters decide how those numbers change after each answer; a challenger
can be FSRS with parameters fitted to the learner's own reviews.

Challengers can also be other kinds of model: a LightGBM classifier predicts
recall from the shared features (`features.py`); it cannot schedule, only
predict in shadow mode.

Which models are active comes from the `memory_models` registry. Fuzzing
(randomly spreading due dates) is off, so scheduling is deterministic: card
state can be rebuilt by replaying the review log, and tests are exact.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import lightgbm
import numpy as np
from fsrs import Card as FsrsCard
from fsrs import Rating, Scheduler, State
from sqlalchemy import Text, cast, func, select
from sqlalchemy.orm import Session

from forglemmigej_api.features import (
    CardInfo,
    PastReview,
    card_features,
    fsrs_scheduler,
    replay_fsrs,
)
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
        return fsrs_scheduler(self.parameters)

    def predict(
        self, history: Sequence[PastReview], at: datetime, card: CardInfo | None = None
    ) -> float | None:
        return predict_from_history(self, history, at)


@dataclass(frozen=True)
class LightGbmModel:
    """A LightGBM recall classifier, stored in the registry as LightGBM's own text
    format plus the ordered feature names it was trained on."""

    version: str
    features: tuple[str, ...]
    booster: lightgbm.Booster

    @classmethod
    def from_parameters(
        cls, version: str, parameters: dict[str, Any]
    ) -> "LightGbmModel":
        booster = lightgbm.Booster(model_str=parameters["model"])
        return cls(version, tuple(parameters["features"]), booster)

    def predict(
        self, history: Sequence[PastReview], at: datetime, card: CardInfo | None = None
    ) -> float | None:
        """Recall probability from the shared feature function; point-in-time like
        every other model (only reviews before `at`)."""
        past = [r for r in history if r.reviewed_at < at]
        if not past:
            return None
        features = card_features(past, at, card)
        row = np.array([[features[name] for name in self.features]], dtype=float)
        return float(self.booster.predict(row, num_threads=1)[0])


type Challenger = FsrsModel | LightGbmModel


class NoChampionError(RuntimeError):
    pass


# Loaded models by (version, digest of the parameters): a LightGBM model is
# parsed once, not on every answer.
_loaded: dict[tuple[str, str], Challenger] = {}


def _load(session: Session, version: str, kind: str, digest: str) -> Challenger:
    key = (version, digest)
    if key not in _loaded:
        parameters = session.scalar(
            select(MemoryModel.parameters).where(MemoryModel.version == version)
        )
        match kind:
            case "fsrs":
                _loaded[key] = FsrsModel(version, tuple(parameters))
            case "lightgbm":
                _loaded[key] = LightGbmModel.from_parameters(version, parameters)
            case _:
                raise ValueError(f"unknown model kind {kind!r} for {version}")
    return _loaded[key]


def active_models(session: Session) -> tuple[FsrsModel, list[Challenger]]:
    """The champion and the challengers, from the model registry."""
    rows = session.execute(
        select(
            MemoryModel.version,
            MemoryModel.kind,
            MemoryModel.role,
            func.md5(cast(MemoryModel.parameters, Text)),
        )
        .where(MemoryModel.role.in_([ModelRole.CHAMPION, ModelRole.CHALLENGER]))
        .order_by(MemoryModel.id)
    ).all()
    champion: FsrsModel | None = None
    challengers: list[Challenger] = []
    for version, kind, role, digest in rows:
        model = _load(session, version, kind, digest)
        if role == ModelRole.CHALLENGER:
            challengers.append(model)
        elif isinstance(model, FsrsModel):
            champion = model
        else:
            raise NoChampionError(f"{version} cannot schedule: only FSRS can")
    if champion is None:
        raise NoChampionError("the model registry has no champion")
    return champion, challengers


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
    card = replay_fsrs(model.parameters, past)
    return float(model.scheduler.get_card_retrievability(card, at.astimezone(UTC)))


def apply_review(
    model: FsrsModel, state: dict[str, Any] | None, grade: Grade, at: datetime
) -> tuple[dict[str, Any], datetime, CardStatus]:
    """New FSRS state, due time and status after a review (champion only)."""
    card = FsrsCard.from_dict(state) if state else FsrsCard()
    card, _log = model.scheduler.review_card(card, Rating(int(grade)), at)
    return card.to_dict(), card.due, _STATUS[card.state]
