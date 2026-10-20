"""Daily review flow: which card comes next, and recording an answer.

Order within a session: due cards first (the memory model decides these), then
new cards up to the daily limit (the curriculum decides these). Every answer is
appended to the review log together with the champion's recall prediction,
computed from the state *before* the answer (point-in-time).
"""

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import Select, and_, case, func, select
from sqlalchemy.orm import Session, aliased

from forglemmigej_api.models import (
    AnswerMode,
    Card,
    CardStatus,
    CardType,
    Grade,
    Item,
    Prediction,
    Review,
)
from forglemmigej_api.scheduling import (
    active_models,
    apply_review,
    predict_from_history,
    predict_from_state,
)
from forglemmigej_api.training.dataset import load_card_history


def start_of_day(now: datetime, timezone: str) -> datetime:
    local = now.astimezone(ZoneInfo(timezone))
    return datetime.combine(local.date(), time.min, tzinfo=local.tzinfo)


def new_words_today(session: Session, now: datetime, timezone: str) -> int:
    """Words introduced today: a word's first card (recognition) counts; its second
    card (cloze or recall) does not, so it never eats into the daily limit."""
    return session.scalar(
        select(func.count())
        .select_from(Card)
        .where(
            Card.introduced_at >= start_of_day(now, timezone),
            Card.card_type == CardType.RECOGNITION,
        )
    )


def _due_cards(now: datetime) -> Select:
    return (
        select(Card)
        .where(
            Card.status.in_([CardStatus.LEARNING, CardStatus.REVIEW]),
            Card.due_at <= now,
        )
        .order_by(Card.due_at, Card.id)
    )


def _new_cards(include_new_words: bool) -> Select:
    """New cards in curriculum order: survival phrases first, then by frequency.
    A word's second card (cloze or recall) only comes once its recognition card
    has graduated to review: right after the first card the answer is still on
    screen. With `include_new_words` false, only such second cards qualify."""
    recognition = aliased(Card)
    query = (
        select(Card)
        .join(Item, Item.id == Card.item_id)
        .outerjoin(
            recognition,
            and_(
                recognition.item_id == Card.item_id,
                recognition.card_type == CardType.RECOGNITION,
            ),
        )
        .where(
            Card.status == CardStatus.NEW,
            (Card.card_type == CardType.RECOGNITION)
            | (recognition.status == CardStatus.REVIEW),
        )
        .order_by(
            case((func.cardinality(Item.topics) > 0, 0), else_=1),
            Item.frequency_rank.asc().nulls_last(),
            case((Card.card_type == CardType.RECOGNITION, 0), else_=1),
            Card.id,
        )
    )
    if not include_new_words:
        query = query.where(Card.card_type != CardType.RECOGNITION)
    return query


@dataclass
class NextCard:
    card: Card | None
    new_remaining: int
    next_due_at: datetime | None  # when the next learning card becomes due


def next_card(
    session: Session, now: datetime, timezone: str, new_per_day: int
) -> NextCard:
    new_remaining = max(0, new_per_day - new_words_today(session, now, timezone))
    card = session.scalars(_due_cards(now).limit(1)).first()
    if card is None:
        card = session.scalars(_new_cards(new_remaining > 0).limit(1)).first()
    next_due_at = None
    if card is None:
        next_due_at = session.scalar(
            select(func.min(Card.due_at)).where(
                Card.status.in_([CardStatus.LEARNING, CardStatus.REVIEW])
            )
        )
    return NextCard(card, new_remaining, next_due_at)


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold().strip()
    return re.sub(r"[\s.,!?;:\"'«»]+", " ", text).strip()


def check_typed_answer(typed: str, expected: str) -> bool:
    return normalize_answer(typed) == normalize_answer(expected)


class AnswerError(ValueError):
    pass


@dataclass
class AnswerResult:
    review: Review
    correct: bool
    predicted_recall: float | None
    card: Card


def answer_card(
    session: Session,
    card_id: int,
    now: datetime,
    *,
    grade: Grade | None = None,
    typed_answer: str | None = None,
    response_time_ms: int | None = None,
    session_id: int | None = None,
) -> AnswerResult:
    """Record an answer: log the review and the champion's prediction, then
    reschedule the card. Cloze cards are graded from the typed answer."""
    card = session.get(Card, card_id, with_for_update=True)
    if card is None or card.status == CardStatus.SUSPENDED:
        raise AnswerError("unknown or suspended card")

    if card.card_type == CardType.CLOZE:
        if typed_answer is None:
            raise AnswerError("cloze cards need a typed answer")
        correct = check_typed_answer(typed_answer, card.target_form or "")
        grade, mode = (Grade.GOOD if correct else Grade.AGAIN), AnswerMode.TYPED
    else:
        if grade is None:
            raise AnswerError("self-graded cards need a grade")
        correct, mode = grade > Grade.AGAIN, AnswerMode.SELF_GRADE

    # Point-in-time: every prediction only uses what was known before this answer.
    champion, challengers = active_models(session)
    predicted = predict_from_state(champion, card.scheduler_state, now)
    shadow: dict[str, float] = {}
    if challengers and predicted is not None:
        history = load_card_history(session, card.id, before=now)
        for challenger in challengers:
            value = predict_from_history(challenger, history, now)
            if value is not None:
                shadow[challenger.version] = value

    review = Review(
        card_id=card.id,
        session_id=session_id,
        reviewed_at=now,
        grade=int(grade),
        response_time_ms=response_time_ms,
        answer_mode=mode,
        typed_answer=typed_answer,
    )
    session.add(review)
    session.flush()
    if predicted is not None:
        session.add(
            Prediction(
                review_id=review.id,
                model_version=champion.version,
                predicted_recall=predicted,
                is_champion=True,
            )
        )
    # Shadow mode: challengers predict every review but do not schedule.
    for version, value in shadow.items():
        session.add(
            Prediction(
                review_id=review.id,
                model_version=version,
                predicted_recall=value,
                is_champion=False,
            )
        )

    card.scheduler_state, card.due_at, card.status = apply_review(
        champion, card.scheduler_state, grade, now
    )
    if card.introduced_at is None:
        card.introduced_at = now
    session.flush()
    return AnswerResult(review, correct, predicted, card)


def due_soon_count(session: Session, now: datetime, within: timedelta) -> int:
    return session.scalar(
        select(func.count())
        .select_from(Card)
        .where(
            Card.status.in_([CardStatus.LEARNING, CardStatus.REVIEW]),
            Card.due_at <= now + within,
        )
    )
