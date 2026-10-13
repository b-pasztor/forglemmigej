from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from forglemmigej_api.content.deck import DeckItem
from forglemmigej_api.content.importer import upsert_items
from forglemmigej_api.content.wiktionary import Lemma
from forglemmigej_api.models import Card, CardType, Item


def deck_item(word: str, rank: int | None, topics: tuple[str, ...] = ()) -> DeckItem:
    lemma = Lemma(word=word, pos="intj", english=f"{word} in English")
    return DeckItem(lemma=lemma, kind="word", frequency_rank=rank, topics=topics)


def test_import_is_idempotent_and_merges_topics(migrated_engine: Engine) -> None:
    with Session(migrated_engine) as session:
        upsert_items(session, [deck_item("hejsa", 95), deck_item("farvel", 314)])
        # The same lemma again as a curated phrase: no rank, but a topic.
        upsert_items(session, [deck_item("hejsa", None, ("greetings",))])
        # Re-running the whole import must not change anything.
        upsert_items(session, [deck_item("hejsa", 95), deck_item("farvel", 314)])

        items = session.scalars(
            select(Item)
            .where(Item.danish.in_(["hejsa", "farvel"]))
            .order_by(Item.danish)
        ).all()
        assert [(i.danish, i.frequency_rank, i.topics) for i in items] == [
            ("farvel", 314, []),
            ("hejsa", 95, ["greetings"]),
        ]
        card_types = session.scalars(
            select(Card.card_type)
            .where(Card.item_id == items[1].id)
            .order_by(Card.card_type)
        ).all()
        assert card_types == [CardType.RECALL, CardType.RECOGNITION]
        assert (
            session.scalar(
                select(func.count())
                .select_from(Card)
                .where(Card.item_id.in_([i.id for i in items]))
            )
            == 4
        )
        session.rollback()
