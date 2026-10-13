"""Database schema.

The review log (`reviews`) is the source of truth: it is append-only and is
what the memory models are trained on. Card scheduling state (`cards.due_at`,
`cards.scheduler_state`) is a cache that can always be rebuilt by replaying
the review log.

Derived values such as "days since last review" or "number of lapses" are
deliberately not stored. The shared feature function computes them from the
review log, so training and serving can never disagree and no information
from after a review can leak into its features.
"""

import enum
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    SmallInteger,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _enum(enum_class: type[enum.Enum], name: str) -> Enum:
    # Stored as text with a CHECK constraint instead of a native Postgres enum,
    # so adding a value later is a simple migration.
    return Enum(
        enum_class,
        name=name,
        native_enum=False,
        create_constraint=True,
        length=32,
        values_callable=lambda members: [m.value for m in members],
    )


class ItemKind(enum.StrEnum):
    WORD = "word"
    SENTENCE = "sentence"


class CardType(enum.StrEnum):
    RECOGNITION = "recognition"  # Danish -> English
    RECALL = "recall"  # English -> Danish
    LISTENING = "listening"  # audio only -> English
    SPEAKING = "speaking"  # English -> spoken Danish


class CardStatus(enum.StrEnum):
    NEW = "new"  # not introduced yet
    LEARNING = "learning"
    REVIEW = "review"
    KNOWN = "known"  # marked as known by the placement test
    SUSPENDED = "suspended"


class AnswerMode(enum.StrEnum):
    SELF_GRADE = "self_grade"
    TYPED = "typed"
    SPOKEN = "spoken"
    PLACEMENT = "placement"


class Grade(enum.IntEnum):
    """FSRS rating scale. The UI currently uses two buttons: AGAIN and GOOD.

    HARD and EASY are kept so a four-button UI can be added without a schema
    change. The ML label is binary: recalled = grade > AGAIN.
    """

    AGAIN = 1
    HARD = 2
    GOOD = 3
    EASY = 4


class Item(Base):
    """Something to learn: a word or a sentence."""

    __tablename__ = "items"
    __table_args__ = (UniqueConstraint("source", "source_ref"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[ItemKind] = mapped_column(_enum(ItemKind, "item_kind"))
    danish: Mapped[str] = mapped_column(Text)
    english: Mapped[str] = mapped_column(Text)
    # Grammatical gender of nouns ("en" / "et"); always learned with the noun.
    article: Mapped[str | None] = mapped_column(Text)
    frequency_rank: Mapped[int | None]
    topics: Mapped[list[str]] = mapped_column(
        ARRAY(Text), server_default=text("'{}'::text[]")
    )
    # Where the item came from (e.g. "tatoeba"), for attribution and licensing.
    source: Mapped[str] = mapped_column(Text)
    source_ref: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Card(Base):
    """One way of practising an item. Each card type is forgotten separately."""

    __tablename__ = "cards"
    __table_args__ = (
        UniqueConstraint("item_id", "card_type"),
        Index("ix_cards_status_due_at", "status", "due_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("items.id"))
    card_type: Mapped[CardType] = mapped_column(_enum(CardType, "card_type"))
    status: Mapped[CardStatus] = mapped_column(
        _enum(CardStatus, "card_status"), server_default=CardStatus.NEW.value
    )
    # Scheduling cache, rebuilt from the review log when needed.
    due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    scheduler_state: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    introduced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class StudySession(Base):
    __tablename__ = "study_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    device: Mapped[str | None] = mapped_column(Text)


class Review(Base):
    """One answer to one card. Append-only: enforced by a database trigger."""

    __tablename__ = "reviews"
    __table_args__ = (
        CheckConstraint("grade BETWEEN 1 AND 4", name="grade_range"),
        CheckConstraint("response_time_ms >= 0", name="response_time_non_negative"),
        Index("ix_reviews_card_id_reviewed_at", "card_id", "reviewed_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    card_id: Mapped[int] = mapped_column(ForeignKey("cards.id"))
    session_id: Mapped[int | None] = mapped_column(ForeignKey("study_sessions.id"))
    reviewed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    grade: Mapped[int] = mapped_column(SmallInteger)
    response_time_ms: Mapped[int | None]
    answer_mode: Mapped[AnswerMode] = mapped_column(_enum(AnswerMode, "answer_mode"))


class Prediction(Base):
    """A model's predicted recall probability for a review.

    Every active model predicts for every review, using only the review
    history before it. The champion's prediction drives scheduling; the
    challengers run in shadow mode. Comparing them on exactly the same
    reviews is what the champion/challenger gate is based on.
    """

    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("review_id", "model_version"),
        CheckConstraint(
            "predicted_recall >= 0 AND predicted_recall <= 1",
            name="predicted_recall_range",
        ),
        # At most one champion prediction per review.
        Index(
            "uq_predictions_one_champion_per_review",
            "review_id",
            unique=True,
            postgresql_where=text("is_champion"),
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    review_id: Mapped[int] = mapped_column(ForeignKey("reviews.id"))
    model_version: Mapped[str] = mapped_column(Text)
    predicted_recall: Mapped[float] = mapped_column(Float)
    is_champion: Mapped[bool]
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
