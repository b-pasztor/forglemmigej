"""Turn review logs into training examples with the shared feature function.

One example per review that has at least one earlier review of the same card:
the first exposure is learning, not a recall test, and the champion makes no
prediction for it either. The label is whether the card was recalled.
"""

from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from forglemmigej_api.features import CardInfo, PastReview, card_features
from forglemmigej_api.models import Card, Item, Review


@dataclass(frozen=True)
class Example:
    card_id: str
    reviewed_at: datetime
    recalled: bool
    features: dict[str, float]
    review_id: int | None = None


def examples_from_history(
    card_id: str,
    reviews: Sequence[PastReview],
    card: CardInfo | None = None,
    review_ids: Sequence[int] | None = None,
    timezone: str = "Europe/Zurich",
) -> Iterator[Example]:
    """Examples for one card. `reviews` must be in chronological order."""
    for i in range(1, len(reviews)):
        target = reviews[i]
        yield Example(
            card_id=card_id,
            reviewed_at=target.reviewed_at,
            recalled=target.recalled,
            # Only reviews[:i]: strictly the past of this review.
            features=card_features(reviews[:i], target.reviewed_at, card, timezone),
            review_id=review_ids[i] if review_ids else None,
        )


def _card_info(card: Card, item: Item) -> CardInfo:
    return CardInfo(
        card_type=str(card.card_type),
        has_context=card.context_item_id is not None,
        word_length=len(item.danish),
        frequency_rank=item.frequency_rank,
    )


def load_card_history(
    session: Session, card_id: int, before: datetime
) -> list[PastReview]:
    """Reviews of one card strictly before `before`, oldest first (serving path)."""
    rows = session.scalars(
        select(Review)
        .where(Review.card_id == card_id, Review.reviewed_at < before)
        .order_by(Review.reviewed_at, Review.id)
    )
    return [PastReview(r.reviewed_at, r.grade, r.response_time_ms) for r in rows]


def load_card_info(session: Session, card_id: int) -> CardInfo:
    card = session.get(Card, card_id)
    return _card_info(card, session.get(Item, card.item_id))


def examples_from_database(
    session: Session, timezone: str = "Europe/Zurich"
) -> list[Example]:
    """All training examples from our own review log (training path)."""
    rows = session.execute(
        select(Review, Card, Item)
        .join(Card, Card.id == Review.card_id)
        .join(Item, Item.id == Card.item_id)
        .order_by(Review.card_id, Review.reviewed_at, Review.id)
    )
    by_card: dict[int, list[tuple[Review, Card, Item]]] = defaultdict(list)
    for review, card, item in rows:
        by_card[card.id].append((review, card, item))

    examples: list[Example] = []
    for card_id, entries in by_card.items():
        _, card, item = entries[0]
        examples.extend(
            examples_from_history(
                str(card_id),
                [
                    PastReview(r.reviewed_at, r.grade, r.response_time_ms)
                    for r, _, _ in entries
                ],
                _card_info(card, item),
                [r.id for r, _, _ in entries],
                timezone,
            )
        )
    return examples


def to_rows(examples: Iterable[Example]) -> list[dict]:
    """Flat dicts, ready for a pandas DataFrame."""
    return [
        {
            "card_id": e.card_id,
            "reviewed_at": e.reviewed_at,
            "recalled": e.recalled,
            **e.features,
        }
        for e in examples
    ]
