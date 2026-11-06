import re
from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from forglemmigej_api.config import Settings, get_settings
from forglemmigej_api.db import get_session
from forglemmigej_api.models import (
    Card,
    CardStatus,
    CardType,
    Grade,
    Item,
    ItemKind,
    Review,
    StudySession,
)
from forglemmigej_api.review import (
    AnswerError,
    answer_card,
    due_soon_count,
    new_words_today,
    next_card,
    start_of_day,
)

app = FastAPI(title="forglemmigej API")

DbSession = Annotated[Session, Depends(get_session)]
AppSettings = Annotated[Settings, Depends(get_settings)]


def utcnow() -> datetime:
    return datetime.now(UTC)


Now = Annotated[datetime, Depends(utcnow)]


class WordOut(BaseModel):
    danish: str
    article: str | None
    english: str
    pos: str | None
    glosses: list[str]
    ipa: str | None
    compound_parts: list[str]


class ContextOut(BaseModel):
    danish: str  # for cloze cards the target is replaced by "___"
    english: str
    target_form: str | None  # hidden (None) on cloze cards until answered


class CardOut(BaseModel):
    card_id: int
    card_type: CardType
    is_new: bool
    word: WordOut
    context: ContextOut | None
    hint: str | None  # first letters of a phrase, e.g. "M____ t__"


class NextOut(BaseModel):
    card: CardOut | None
    new_remaining: int
    practice_waiting: int
    due_now: int
    next_due_at: datetime | None


class AnswerIn(BaseModel):
    grade: Literal[1, 3] | None = (
        None  # Again / Good; cloze cards are graded from the text
    )
    typed_answer: str | None = Field(default=None, max_length=200)
    response_time_ms: int | None = Field(default=None, ge=0)
    session_id: int | None = None


class AnswerOut(BaseModel):
    correct: bool
    grade: int
    expected: str
    full_sentence: str | None
    next_due_at: datetime | None
    predicted_recall: float | None


class SessionIn(BaseModel):
    device: str | None = Field(default=None, max_length=200)


class SessionOut(BaseModel):
    id: int


class StatsOut(BaseModel):
    reviewed_today: int
    correct_today: int
    new_today: int
    due_now: int
    due_next_7_days: int
    cards_by_status: dict[str, int]


def mask(sentence: str, form: str) -> str:
    pattern = rf"(?<![\wæøå]){re.escape(form)}(?![\wæøå])"
    return re.sub(pattern, "___", sentence, count=1, flags=re.IGNORECASE)


def first_letters(phrase: str) -> str:
    """A phrase with only its first letters: "Mange tak" -> "M____ t__"."""
    return " ".join(word[0] + "_" * (len(word) - 1) for word in phrase.split())


def phrase_hint(card: Card, item: Item) -> str | None:
    """Several phrases can fit one translation ("Thanks": tak, mange tak...). The
    learner cannot know which is asked for, so a miss would say nothing about their
    memory. First letters identify the phrase; single words have a sentence for that."""
    if item.kind != ItemKind.PHRASE:
        return None
    if card.card_type == CardType.CLOZE:
        phrase = card.target_form or ""
    elif card.card_type == CardType.RECALL:
        phrase = item.danish
    else:
        return None
    return first_letters(phrase) if " " in phrase.strip() else None


def card_out(session: Session, card: Card) -> CardOut:
    item = session.get(Item, card.item_id)
    details = item.details or {}
    context = None
    if card.context_item_id is not None:
        sentence = session.get(Item, card.context_item_id)
        is_cloze = card.card_type == CardType.CLOZE
        context = ContextOut(
            danish=mask(sentence.danish, card.target_form)
            if is_cloze
            else sentence.danish,
            english=sentence.english,
            target_form=None if is_cloze else card.target_form,
        )
    return CardOut(
        card_id=card.id,
        card_type=card.card_type,
        is_new=card.status == CardStatus.NEW,
        hint=phrase_hint(card, item),
        word=WordOut(
            danish=item.danish,
            article=item.article,
            english=item.english,
            pos=details.get("pos"),
            glosses=details.get("glosses", []),
            ipa=details.get("ipa"),
            compound_parts=details.get("compound_parts", []),
        ),
        context=context,
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/sessions")
def start_session(body: SessionIn, db: DbSession, now: Now) -> SessionOut:
    study_session = StudySession(started_at=now, device=body.device)
    db.add(study_session)
    db.flush()
    return SessionOut(id=study_session.id)


@app.post("/api/sessions/{session_id}/end")
def end_session(session_id: int, db: DbSession, now: Now) -> SessionOut:
    study_session = db.get(StudySession, session_id)
    if study_session is None:
        raise HTTPException(404, "unknown session")
    study_session.ended_at = now
    return SessionOut(id=session_id)


@app.get("/api/next")
def get_next(db: DbSession, now: Now, settings: AppSettings) -> NextOut:
    result = next_card(db, now, settings.timezone, settings.new_words_per_day)
    return NextOut(
        card=card_out(db, result.card) if result.card else None,
        new_remaining=result.new_remaining,
        practice_waiting=result.practice_waiting,
        due_now=due_soon_count(db, now, timedelta(0)),
        next_due_at=result.next_due_at,
    )


@app.post("/api/cards/{card_id}/answer")
def answer(card_id: int, body: AnswerIn, db: DbSession, now: Now) -> AnswerOut:
    try:
        result = answer_card(
            db,
            card_id,
            now,
            grade=Grade(body.grade) if body.grade else None,
            typed_answer=body.typed_answer,
            response_time_ms=body.response_time_ms,
            session_id=body.session_id,
        )
    except AnswerError as error:
        raise HTTPException(422, str(error)) from error
    card = result.card
    item = db.get(Item, card.item_id)
    sentence = db.get(Item, card.context_item_id) if card.context_item_id else None
    return AnswerOut(
        correct=result.correct,
        grade=result.review.grade,
        expected=card.target_form or item.danish,
        full_sentence=sentence.danish if sentence else None,
        next_due_at=card.due_at,
        predicted_recall=result.predicted_recall,
    )


@app.get("/api/stats")
def stats(db: DbSession, now: Now, settings: AppSettings) -> StatsOut:
    today = start_of_day(now, settings.timezone)
    reviewed, correct = db.execute(
        select(func.count(), func.count().filter(Review.grade > Grade.AGAIN)).where(
            Review.reviewed_at >= today
        )
    ).one()
    by_status = dict(
        db.execute(select(Card.status, func.count()).group_by(Card.status)).all()
    )
    return StatsOut(
        reviewed_today=reviewed,
        correct_today=correct,
        new_today=new_words_today(db, now, settings.timezone),
        due_now=due_soon_count(db, now, timedelta(0)),
        due_next_7_days=due_soon_count(db, now, timedelta(days=7)),
        cards_by_status={str(k): v for k, v in by_status.items()},
    )
