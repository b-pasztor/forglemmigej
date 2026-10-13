from datetime import UTC, datetime

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from forglemmigej_api.content.deck import DeckItem
from forglemmigej_api.content.examples import Example
from forglemmigej_api.content.importer import plan_cards, prune, upsert_deck
from forglemmigej_api.content.tatoeba import Sentence, Translation
from forglemmigej_api.content.wiktionary import Lemma
from forglemmigej_api.models import (
    AnswerMode,
    Card,
    CardStatus,
    CardType,
    Grade,
    Item,
    Review,
)


def deck_item(
    word: str, rank: int | None, topics: tuple[str, ...] = (), pos: str = "noun"
) -> DeckItem:
    lemma = Lemma(
        word=word, pos=pos, english=f"{word} in English", glosses=(f"{word}-en",)
    )
    return DeckItem(lemma=lemma, kind="word", frequency_rank=rank, topics=topics)


def example(sid: str, text: str, form: str) -> Example:
    translation = Translation(f"e{sid}", "English")
    return Example(Sentence(sid, text, (translation,)), translation, form, "x")


def cards_of(session: Session, word: str) -> dict[CardType, Card]:
    rows = session.scalars(
        select(Card).join(Item, Card.item_id == Item.id).where(Item.danish == word)
    )
    return {card.card_type: card for card in rows}


def test_plan_cards() -> None:
    noun = deck_item("hus", 1)
    with_sentence = plan_cards(noun, example("1", "Huset er stort.", "Huset"))
    assert [p.card_type for p in with_sentence] == [
        CardType.RECOGNITION,
        CardType.CLOZE,
    ]
    assert [p.card_type for p in plan_cards(noun, None)] == [
        CardType.RECOGNITION,
        CardType.RECALL,
    ]
    assert plan_cards(deck_item("et", 1, pos="article"), None) == []
    curated = DeckItem(
        lemma=Lemma(word="skål", pos="intj", english="cheers", glosses=("cheers",)),
        kind="phrase",
        frequency_rank=None,
    )
    assert [p.card_type for p in plan_cards(curated, None)] == [
        CardType.RECOGNITION,
        CardType.RECALL,
    ]


def test_import_is_idempotent_and_merges_topics(migrated_engine: Engine) -> None:
    with Session(migrated_engine) as session:
        upsert_deck(session, [deck_item("hejsa", 95), deck_item("farvel", 314)])
        # The same lemma again as a curated phrase: no rank, but a topic.
        upsert_deck(session, [deck_item("hejsa", None, ("greetings",))])
        # Re-running the whole import must not change anything.
        upsert_deck(session, [deck_item("hejsa", 95), deck_item("farvel", 314)])

        items = session.scalars(
            select(Item)
            .where(Item.danish.in_(["hejsa", "farvel"]))
            .order_by(Item.danish)
        ).all()
        assert [(i.danish, i.frequency_rank, i.topics) for i in items] == [
            ("farvel", 314, []),
            ("hejsa", 95, ["greetings"]),
        ]
        assert set(cards_of(session, "hejsa")) == {
            CardType.RECOGNITION,
            CardType.RECALL,
        }
        session.rollback()


def test_example_sentence_becomes_card_context(migrated_engine: Engine) -> None:
    with Session(migrated_engine) as session:
        hus = deck_item("huus", 1)
        upsert_deck(
            session,
            [hus],
            {hus.lemma.key: example("900", "Huuset er stort.", "Huuset")},
        )
        cards = cards_of(session, "huus")
        assert set(cards) == {CardType.RECOGNITION, CardType.CLOZE}
        cloze = cards[CardType.CLOZE]
        context = session.get(Item, cloze.context_item_id)
        assert (context.kind, context.danish, context.source) == (
            "sentence",
            "Huuset er stort.",
            "tatoeba",
        )
        assert cloze.target_form == "Huuset"
        session.rollback()


def test_prune_deletes_unreviewed_and_suspends_reviewed(
    migrated_engine: Engine,
) -> None:
    with Session(migrated_engine) as session:
        planned = upsert_deck(
            session,
            [deck_item("beholde", 1), deck_item("slette", 2), deck_item("pause", 3)],
        )
        session.add(
            Review(
                card_id=cards_of(session, "pause")[CardType.RECOGNITION].id,
                reviewed_at=datetime.now(UTC),
                grade=Grade.GOOD,
                answer_mode=AnswerMode.SELF_GRADE,
            )
        )
        session.flush()

        # "slette" and "pause" leave the deck; everything else stays planned.
        gone = set(
            session.scalars(select(Item.id).where(Item.danish.in_(["slette", "pause"])))
        )
        keep = {
            item_id: types for item_id, types in planned.items() if item_id not in gone
        }
        deleted, suspended = prune(session, keep)

        # slette: 2 cards deleted; pause: recall deleted, reviewed recognition suspended.
        assert (deleted, suspended) == (3, 1)
        assert session.scalar(select(Item).where(Item.danish == "slette")) is None
        pause = cards_of(session, "pause")
        assert {t: c.status for t, c in pause.items()} == {
            CardType.RECOGNITION: CardStatus.SUSPENDED
        }
        assert set(cards_of(session, "beholde")) == {
            CardType.RECOGNITION,
            CardType.RECALL,
        }
        session.rollback()
