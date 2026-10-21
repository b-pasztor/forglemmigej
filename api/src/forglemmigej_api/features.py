"""The shared feature function: one card's review history -> model features.

Training and serving both call `card_features`, so the model sees exactly the
same numbers in both places (no train/serve skew). It is point-in-time by
construction: it receives only the reviews *before* the moment being predicted
and refuses anything later, so information from the future cannot leak in.

Features come in two groups:
- HISTORY_FEATURES need only the review history. Public review logs (used for
  pre-training) have these too.
- CARD_FEATURES describe the card and the moment; only our own data has them.
"""

import math
from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import lru_cache
from zoneinfo import ZoneInfo

from fsrs import Card as FsrsCard
from fsrs import Rating, Scheduler

from forglemmigej_api.models import Grade

SECONDS_PER_DAY = 86_400.0

HISTORY_FEATURES = (
    "elapsed_days",  # since the last review: the main driver of forgetting
    "log_elapsed_days",
    "n_reviews",
    "n_lapses",  # how often it was forgotten
    "last_recalled",
    "recall_rate",
    "success_streak",  # recalled in a row, up to the last review
    "prev_interval_days",  # gap between the last two reviews
    "days_since_first",
    "log_last_response_s",  # slow answers suggest a weaker memory; NaN if unknown
    "fsrs_recall",  # FSRS's own prediction, replayed from the history
)
CARD_FEATURES = (
    "is_cloze",
    "is_recall",
    "has_context",
    "word_length",
    "log_frequency_rank",  # NaN for curated phrases without a rank
    "hour_of_day",
)


@dataclass(frozen=True)
class PastReview:
    reviewed_at: datetime
    grade: int  # FSRS scale, 1..4
    response_time_ms: int | None = None

    @property
    def recalled(self) -> bool:
        return self.grade > Grade.AGAIN


@dataclass(frozen=True)
class CardInfo:
    card_type: str
    has_context: bool
    word_length: int
    frequency_rank: int | None


class FutureLeakError(ValueError):
    """A history entry is not strictly before the prediction time."""


# FSRS-6 default parameters (fsrs 6.3.2), pinned: models are trained on the
# `fsrs_recall` feature, so a library upgrade must not change it silently. The
# same values as the `fsrs6-default` champion seeded by the registry migration.
FSRS6_DEFAULT_PARAMETERS = (
    0.212, 1.2931, 2.3065, 8.2956, 6.4133, 0.8334, 3.0194, 0.001, 1.8722, 0.1666, 0.796,
    1.4835, 0.0614, 0.2629, 1.6483, 0.6014, 1.8729, 0.5425, 0.0912, 0.0658, 0.1542,
)  # fmt: skip


@lru_cache(maxsize=16)
def fsrs_scheduler(parameters: tuple[float, ...]) -> Scheduler:
    """FSRS with the given parameters; fuzzing off, so replays are exact."""
    return Scheduler(parameters=parameters, enable_fuzzing=False)


@lru_cache(maxsize=200_000)
def replay_fsrs(
    parameters: tuple[float, ...], reviews: tuple[tuple[datetime, int], ...]
) -> FsrsCard:
    """FSRS card state after replaying `reviews` (UTC times, oldest first). Cached
    per prefix, so a history that grows by one review costs one FSRS step."""
    card = replay_fsrs(parameters, reviews[:-1]) if len(reviews) > 1 else FsrsCard()
    reviewed_at, grade = reviews[-1]
    card, _ = fsrs_scheduler(parameters).review_card(
        copy(card), Rating(grade), reviewed_at
    )
    return card


def _fsrs_recall(history: Sequence[PastReview], at: datetime) -> float:
    state = replay_fsrs(
        FSRS6_DEFAULT_PARAMETERS, tuple((r.reviewed_at, r.grade) for r in history)
    )
    return float(
        fsrs_scheduler(FSRS6_DEFAULT_PARAMETERS).get_card_retrievability(state, at)
    )


def _days(later: datetime, earlier: datetime) -> float:
    return (later - earlier).total_seconds() / SECONDS_PER_DAY


def history_features(history: Sequence[PastReview], at: datetime) -> dict[str, float]:
    if not history:
        raise ValueError("a card needs at least one past review to be predicted")
    # Database timestamps carry their own tz objects; FSRS insists on exactly UTC.
    at = at.astimezone(UTC)
    history = [replace(r, reviewed_at=r.reviewed_at.astimezone(UTC)) for r in history]
    if any(r.reviewed_at >= at for r in history):
        raise FutureLeakError(
            "history must only contain reviews before the prediction time"
        )
    history = sorted(history, key=lambda r: r.reviewed_at)
    last = history[-1]
    recalled = [r.recalled for r in history]
    streak = 0
    for ok in reversed(recalled):
        if not ok:
            break
        streak += 1
    elapsed = _days(at, last.reviewed_at)
    response = last.response_time_ms
    return {
        "elapsed_days": elapsed,
        "log_elapsed_days": math.log1p(elapsed),
        "n_reviews": float(len(history)),
        "n_lapses": float(sum(not ok for ok in recalled)),
        "last_recalled": float(recalled[-1]),
        "recall_rate": sum(recalled) / len(recalled),
        "success_streak": float(streak),
        "prev_interval_days": _days(last.reviewed_at, history[-2].reviewed_at)
        if len(history) > 1
        else 0.0,
        "days_since_first": _days(at, history[0].reviewed_at),
        "log_last_response_s": math.log1p(response / 1000)
        if response is not None
        else math.nan,
        "fsrs_recall": _fsrs_recall(history, at),
    }


def card_features(
    history: Sequence[PastReview],
    at: datetime,
    card: CardInfo | None = None,
    timezone: str = "Europe/Zurich",
) -> dict[str, float]:
    """All features for predicting a review of one card at `at`."""
    features = history_features(history, at)
    if card is not None:
        features |= {
            "is_cloze": float(card.card_type == "cloze"),
            "is_recall": float(card.card_type == "recall"),
            "has_context": float(card.has_context),
            "word_length": float(card.word_length),
            "log_frequency_rank": math.log(card.frequency_rank)
            if card.frequency_rank
            else math.nan,
            "hour_of_day": float(at.astimezone(ZoneInfo(timezone)).hour),
        }
    return features
