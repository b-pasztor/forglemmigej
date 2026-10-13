"""The champion memory model: FSRS.

FSRS keeps two numbers per card (stability and difficulty) and from them
predicts the probability of recall at any moment ("retrievability"). A card is
due when that probability would drop below the desired retention (90%).

Fuzzing (randomly spreading due dates) is off, so scheduling is deterministic:
card state can be rebuilt by replaying the review log, and tests are exact.
"""

from datetime import datetime
from importlib.metadata import version
from typing import Any

from fsrs import Card as FsrsCard
from fsrs import Rating, Scheduler, State

from forglemmigej_api.models import CardStatus, Grade

CHAMPION_MODEL = f"fsrs-{version('fsrs')}-default"

_scheduler = Scheduler(enable_fuzzing=False)

_STATUS = {
    State.Learning: CardStatus.LEARNING,
    State.Review: CardStatus.REVIEW,
    State.Relearning: CardStatus.LEARNING,
}


def predict_recall(state: dict[str, Any] | None, at: datetime) -> float | None:
    """Probability that the card is recalled at `at`, or None before its first review
    (the first exposure is learning, not a recall test)."""
    if not state:
        return None
    card = FsrsCard.from_dict(state)
    if card.last_review is None:
        return None
    return float(_scheduler.get_card_retrievability(card, at))


def apply_review(
    state: dict[str, Any] | None, grade: Grade, at: datetime
) -> tuple[dict[str, Any], datetime, CardStatus]:
    """New FSRS state, due time and status after a review."""
    card = FsrsCard.from_dict(state) if state else FsrsCard()
    card, _log = _scheduler.review_card(card, Rating(int(grade)), at)
    return card.to_dict(), card.due, _STATUS[card.state]
