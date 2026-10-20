from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from forglemmigej_api.app import app, mask, utcnow
from forglemmigej_api.config import get_settings
from forglemmigej_api.db import get_session
from forglemmigej_api.models import (
    Card,
    CardStatus,
    CardType,
    Item,
    ItemKind,
    Prediction,
    Review,
)


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 14, 8, 0, tzinfo=UTC)

    def advance(self, **kwargs: float) -> None:
        self.now += timedelta(**kwargs)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def client(db: Session, clock: Clock) -> Iterator[TestClient]:
    # Only this test's cards: start from an empty deck inside the rolled-back transaction.
    db.execute(delete(Card).where(~Card.id.in_(select(Review.card_id))))
    settings = get_settings().model_copy(update={"new_words_per_day": 3})
    app.dependency_overrides[get_session] = lambda: db
    app.dependency_overrides[utcnow] = lambda: clock.now
    app.dependency_overrides[get_settings] = lambda: settings
    yield TestClient(app)
    app.dependency_overrides.clear()


def add_word(
    db: Session,
    danish: str,
    rank: int,
    sentence: tuple[str, str, str] | None = None,
    topics: list[str] | None = None,
) -> dict[CardType, int]:
    item = Item(
        kind=ItemKind.WORD,
        danish=danish,
        english=f"{danish}-en",
        frequency_rank=rank,
        topics=topics or [],
        source="test",
        source_ref=f"{danish}|test",
    )
    db.add(item)
    db.flush()
    if sentence is None:
        cards = [
            Card(item_id=item.id, card_type=t)
            for t in (CardType.RECOGNITION, CardType.RECALL)
        ]
    else:
        text, english, form = sentence
        context = Item(
            kind=ItemKind.SENTENCE,
            danish=text,
            english=english,
            source="test-s",
            source_ref=text,
        )
        db.add(context)
        db.flush()
        cards = [
            Card(
                item_id=item.id,
                card_type=t,
                context_item_id=context.id,
                target_form=form,
            )
            for t in (CardType.RECOGNITION, CardType.CLOZE)
        ]
    db.add_all(cards)
    db.flush()
    return {c.card_type: c.id for c in cards}


def test_mask_replaces_only_the_target_word() -> None:
    assert mask("Huset er hus.", "Huset") == "___ er hus."
    assert mask("Er det her?", "er") == "___ det her?"


def test_new_cards_follow_the_curriculum(client: TestClient, db: Session) -> None:
    add_word(db, "hus", 300)
    add_word(db, "jeg", 1)
    add_word(db, "hej", 95, topics=["greetings"])
    card = client.get("/api/next").json()["card"]
    assert (card["word"]["danish"], card["card_type"], card["is_new"]) == (
        "hej",
        "recognition",
        True,
    )


def test_first_answer_logs_review_without_prediction(
    client: TestClient, db: Session, clock: Clock
) -> None:
    ids = add_word(db, "hus", 1)
    response = client.post(
        f"/api/cards/{ids[CardType.RECOGNITION]}/answer",
        json={"grade": 3, "response_time_ms": 1200},
    )
    assert response.status_code == 200
    assert response.json()["predicted_recall"] is None

    card = db.get(Card, ids[CardType.RECOGNITION])
    assert card.status == CardStatus.LEARNING
    assert card.introduced_at == clock.now
    assert card.due_at > clock.now
    review = db.scalars(select(Review).where(Review.card_id == card.id)).one()
    assert (review.grade, review.response_time_ms, review.answer_mode) == (
        3,
        1200,
        "self_grade",
    )


def test_later_answer_stores_the_champion_prediction(
    client: TestClient, db: Session, clock: Clock
) -> None:
    ids = add_word(db, "hus", 1)
    url = f"/api/cards/{ids[CardType.RECOGNITION]}/answer"
    client.post(url, json={"grade": 3})
    clock.advance(minutes=15)
    assert (
        client.get("/api/next").json()["card"]["card_id"] == ids[CardType.RECOGNITION]
    )

    predicted = client.post(url, json={"grade": 3}).json()["predicted_recall"]
    assert 0 < predicted <= 1
    prediction = db.scalars(select(Prediction)).one()
    assert (prediction.model_version, prediction.is_champion) == ("fsrs6-default", True)
    assert prediction.predicted_recall == pytest.approx(predicted)


def graduate_recognition(client: TestClient, clock: Clock, card_id: int) -> None:
    """Two good answers move a new card through the learning steps into review."""
    client.post(f"/api/cards/{card_id}/answer", json={"grade": 3})
    clock.advance(minutes=15)
    client.post(f"/api/cards/{card_id}/answer", json={"grade": 3})


def test_cloze_comes_after_recognition_graduates_and_is_graded_from_text(
    client: TestClient, db: Session, clock: Clock
) -> None:
    ids = add_word(db, "hus", 1, ("Huset er stort.", "The house is big.", "Huset"))
    first = client.get("/api/next").json()["card"]
    assert first["card_type"] == "recognition"
    assert first["context"] == {
        "danish": "Huset er stort.",
        "english": "The house is big.",
        "target_form": "Huset",
    }
    client.post(f"/api/cards/{ids[CardType.RECOGNITION]}/answer", json={"grade": 3})
    # Still learning: the sentence was just on screen, so no cloze yet.
    assert client.get("/api/next").json()["card"] is None

    clock.advance(minutes=15)
    client.post(f"/api/cards/{ids[CardType.RECOGNITION]}/answer", json={"grade": 3})
    cloze = client.get("/api/next").json()["card"]
    assert cloze["card_type"] == "cloze"
    assert cloze["context"] == {
        "danish": "___ er stort.",
        "english": "The house is big.",
        "target_form": None,
    }

    url = f"/api/cards/{ids[CardType.CLOZE]}/answer"
    assert client.post(url, json={"grade": 3}).status_code == 422  # must type
    result = client.post(url, json={"typed_answer": " huset! "}).json()
    assert (result["correct"], result["grade"], result["expected"]) == (
        True,
        3,
        "Huset",
    )
    assert result["full_sentence"] == "Huset er stort."


def test_wrong_typed_answer_is_graded_again(
    client: TestClient, db: Session, clock: Clock
) -> None:
    ids = add_word(db, "hus", 1, ("Huset er stort.", "The house is big.", "Huset"))
    graduate_recognition(client, clock, ids[CardType.RECOGNITION])
    result = client.post(
        f"/api/cards/{ids[CardType.CLOZE]}/answer", json={"typed_answer": "hus"}
    ).json()
    assert (result["correct"], result["grade"]) == (False, 1)
    review = db.scalars(
        select(Review).where(Review.card_id == ids[CardType.CLOZE])
    ).one()
    assert review.typed_answer == "hus"


def test_daily_new_card_limit(client: TestClient, db: Session, clock: Clock) -> None:
    for rank, word in enumerate(["en", "to", "tre", "fire"], 1):
        add_word(db, word, rank)
    for _ in range(3):
        card = client.get("/api/next").json()["card"]
        client.post(f"/api/cards/{card['card_id']}/answer", json={"grade": 3})
    body = client.get("/api/next").json()
    assert body["card"] is None
    assert body["new_remaining"] == 0
    assert body["next_due_at"] is not None  # learning cards come back later today

    clock.advance(days=1)
    assert client.get("/api/next").json()["card"] is not None


def test_stats(client: TestClient, db: Session) -> None:
    ids = add_word(db, "hus", 1)
    client.post(f"/api/cards/{ids[CardType.RECOGNITION]}/answer", json={"grade": 1})
    stats = client.get("/api/stats").json()
    assert (stats["reviewed_today"], stats["correct_today"], stats["new_today"]) == (
        1,
        0,
        1,
    )


def test_second_cards_do_not_use_the_new_word_limit(
    client: TestClient, db: Session, clock: Clock
) -> None:
    words = [
        add_word(db, word, rank) for rank, word in enumerate(["en", "to", "tre"], 1)
    ]
    for _ in range(3):  # the daily limit in these tests is 3 new words
        card = client.get("/api/next").json()["card"]
        assert card["card_type"] == "recognition"
        client.post(f"/api/cards/{card['card_id']}/answer", json={"grade": 3})
    assert client.get("/api/next").json()["new_remaining"] == 0

    # Graduate the first word's recognition card; its recall card unlocks and
    # comes even though no new words are left today.
    clock.advance(minutes=15)
    client.post(
        f"/api/cards/{words[0][CardType.RECOGNITION]}/answer", json={"grade": 3}
    )
    clock.advance(minutes=1)
    for _ in range(4):
        card = client.get("/api/next").json()["card"]
        if card is None or card["card_id"] == words[0][CardType.RECALL]:
            break
        client.post(f"/api/cards/{card['card_id']}/answer", json={"grade": 3})
    assert card is not None and card["card_id"] == words[0][CardType.RECALL]
    assert card["card_type"] == "recall"
