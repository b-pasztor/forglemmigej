from datetime import UTC, datetime

from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from forglemmigej_api.content.deck import DeckItem
from forglemmigej_api.content.importer import prune_items, upsert_items
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


def test_prune_deletes_unreviewed_and_suspends_reviewed(
    migrated_engine: Engine,
) -> None:
    with Session(migrated_engine) as session:
        keep, unreviewed, reviewed = (
            deck_item("beholde", 1),
            deck_item("slette", 2),
            deck_item("pause", 3),
        )
        upsert_items(session, [keep, unreviewed, reviewed])
        reviewed_card = session.scalars(
            select(Card).join(Item).where(Item.danish == "pause")
        ).first()
        session.add(
            Review(
                card_id=reviewed_card.id,
                reviewed_at=datetime.now(UTC),
                grade=Grade.GOOD,
                answer_mode=AnswerMode.SELF_GRADE,
            )
        )
        session.flush()

        # Other test data in the same database must not be touched: prune only
        # compares against what we pass in, so include everything except two.
        others = session.scalars(
            select(Item.source_ref).where(
                Item.source == "wiktionary",
                Item.danish.not_in(["slette", "pause"]),
            )
        ).all()
        keep_all = [deck_item(ref.split("|")[0], None) for ref in others]
        assert prune_items(session, keep_all) == (1, 1)

        assert session.scalar(select(Item).where(Item.danish == "slette")) is None
        statuses = session.scalars(
            select(Card.status).join(Item).where(Item.danish == "pause")
        ).all()
        assert set(statuses) == {CardStatus.SUSPENDED}
        session.rollback()
