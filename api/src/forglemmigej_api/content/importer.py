"""Import the learning deck into the database.

Usage (from api/):
    uv run python -m forglemmigej_api.content.importer --words 500
    uv run python -m forglemmigej_api.content.importer --words 300 --dry-run

The import is idempotent: items are upserted by (source, source_ref), and the
cards each item should have are created or updated. Words with a validated
example sentence get recognition + cloze cards in that sentence; unambiguous
words without one get plain recognition + recall cards; the rest get no cards.

Anything that is no longer part of the plan (e.g. after a curation fix) is
deleted, or suspended if it has already been reviewed, so the review log stays
intact. Reviewed cards also keep their sentence.
"""

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import (
    and_,
    delete,
    exists,
    func,
    literal_column,
    or_,
    select,
    tuple_,
    update,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from forglemmigej_api.content.deck import (
    Curation,
    DeckItem,
    build_phrases,
    build_words,
    read_frequency_list,
)
from forglemmigej_api.content.examples import Example, select_examples
from forglemmigej_api.content.tatoeba import load_corpus
from forglemmigej_api.content.wiktionary import Dictionary
from forglemmigej_api.models import Card, CardStatus, CardType, Item, Review

REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE = "wiktionary"
SENTENCE_SOURCE = "tatoeba"
# Parts of speech whose meaning is usually clear without a sentence.
CONTENT_POS = frozenset({"noun", "verb", "adj", "num"})


@dataclass(frozen=True)
class CardPlan:
    card_type: CardType
    sentence: Example | None = None


def plan_cards(item: DeckItem, example: Example | None) -> list[CardPlan]:
    if example:
        return [
            CardPlan(CardType.RECOGNITION, example),
            CardPlan(CardType.CLOZE, example),
        ]
    lemma = item.lemma
    # Curated phrases ("skål", "mange tak") have one clear meaning.
    unambiguous = item.kind == "phrase" or (
        lemma.pos in CONTENT_POS and len(lemma.glosses) <= 2
    )
    if unambiguous:
        return [CardPlan(CardType.RECOGNITION), CardPlan(CardType.RECALL)]
    return []  # e.g. "en", "et", "jo": learned through sentences and phrases instead


def item_values(item: DeckItem) -> dict:
    lemma = item.lemma
    return {
        "kind": "phrase" if " " in lemma.word else "word",
        "danish": lemma.word,
        "english": lemma.english,
        "article": lemma.article,
        "frequency_rank": item.frequency_rank,
        "topics": list(item.topics),
        "source": SOURCE,
        "source_ref": lemma.key,
        "details": {
            "pos": lemma.pos,
            "glosses": list(lemma.glosses),
            "ipa": lemma.ipa,
            "compound_parts": list(lemma.compound_parts),
            "seen_as": item.seen_as,
        },
    }


def _upsert_item(session: Session, values: dict, merge_rank_and_topics: bool) -> int:
    stmt = insert(Item).values(**values)
    excluded = stmt.excluded
    updates = {
        "kind": excluded.kind,
        "danish": excluded.danish,
        "english": excluded.english,
        "article": excluded.article,
        "details": excluded.details,
    }
    if merge_rank_and_topics:
        # A phrase can also be a frequent word ("hej"): keep the rank, merge topics.
        updates["frequency_rank"] = func.coalesce(
            excluded.frequency_rank, Item.frequency_rank
        )
        updates["topics"] = func.coalesce(
            select(func.array_agg(literal_column("t").distinct()))
            .select_from(func.unnest(Item.topics.concat(excluded.topics)).alias("t"))
            .scalar_subquery(),
            literal_column("'{}'::text[]"),
        )
    stmt = stmt.on_conflict_do_update(
        index_elements=["source", "source_ref"], set_=updates
    )
    return session.execute(stmt.returning(Item.id)).scalar_one()


def _upsert_sentence(session: Session, example: Example) -> int:
    return _upsert_item(
        session,
        {
            "kind": "sentence",
            "danish": example.sentence.text,
            "english": example.translation.text,
            "source": SENTENCE_SOURCE,
            "source_ref": example.sentence.id,
            "details": {"translation_id": example.translation.id},
        },
        merge_rank_and_topics=False,
    )


def _has_reviews():
    return exists().where(Review.card_id == Card.id)


def upsert_deck(
    session: Session,
    items: Sequence[DeckItem],
    examples: Mapping[str, Example] | None = None,
) -> dict[int, list[CardType]]:
    """Upsert items, their example sentences and planned cards.

    Returns item id -> planned card types (used for pruning)."""
    examples = examples or {}
    planned: dict[int, list[CardType]] = {}
    for item in items:
        item_id = _upsert_item(session, item_values(item), merge_rank_and_topics=True)
        plans = plan_cards(item, examples.get(item.lemma.key))
        planned[item_id] = [plan.card_type for plan in plans]
        for plan in plans:
            context_id = (
                _upsert_sentence(session, plan.sentence) if plan.sentence else None
            )
            target = plan.sentence.target_form if plan.sentence else None
            stmt = insert(Card).values(
                item_id=item_id,
                card_type=plan.card_type,
                context_item_id=context_id,
                target_form=target,
            )
            session.execute(
                stmt.on_conflict_do_update(
                    index_elements=["item_id", "card_type"],
                    set_={"context_item_id": context_id, "target_form": target},
                    # A reviewed card keeps the sentence it was learned with.
                    where=~_has_reviews(),
                )
            )
    return planned


def prune(
    session: Session, planned: Mapping[int, Sequence[CardType]]
) -> tuple[int, int]:
    """Remove cards and items that are no longer planned. Returns (deleted, suspended) cards."""
    wanted = [(item_id, t.value) for item_id, types in planned.items() for t in types]
    stale_card = and_(
        Card.item_id.in_(select(Item.id).where(Item.source == SOURCE)),
        tuple_(Card.item_id, Card.card_type).not_in(wanted) if wanted else True,
    )
    suspended = session.execute(
        update(Card)
        .where(stale_card, _has_reviews(), Card.status != CardStatus.SUSPENDED)
        .values(status=CardStatus.SUSPENDED)
    ).rowcount
    deleted = session.execute(delete(Card).where(stale_card, ~_has_reviews())).rowcount

    # Items: wiktionary items that left the deck and have no cards left; sentences no card uses.
    session.execute(
        delete(Item).where(
            or_(
                and_(Item.source == SOURCE, Item.id.not_in(list(planned))),
                and_(
                    Item.source == SENTENCE_SOURCE,
                    Item.id.not_in(
                        select(Card.context_item_id).where(
                            Card.context_item_id.is_not(None)
                        )
                    ),
                ),
            ),
            ~exists().where(Card.item_id == Item.id),
        )
    )
    return deleted, suspended


def format_item(item: DeckItem, example: Example | None) -> str:
    lemma = item.lemma
    head = f"{lemma.article} {lemma.word}" if lemma.article else lemma.word
    rank = f"#{item.frequency_rank}" if item.frequency_rank else "phrase"
    cards = "+".join(p.card_type.value for p in plan_cards(item, example)) or "no cards"
    line = f"{rank:>7}  {head:<22} {lemma.pos:<7} {lemma.english[:45]:<45} [{cards}]"
    if example:
        line += f"\n{'':9}{example.sentence.text}  ->  {example.translation.text}"
    return line


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--words", type=int, default=500, help="number of frequent words"
    )
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument(
        "--curation-dir", type=Path, default=REPO_ROOT / "data" / "curation"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the deck, do not write"
    )
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    dictionary = Dictionary.from_jsonl(args.raw_dir / "kaikki-danish.jsonl")
    curation = Curation.load(args.curation_dir)
    with (args.raw_dir / "da_50k.txt").open(encoding="utf-8") as f:
        ranked = list(read_frequency_list(f))
    words = build_words(dictionary, ranked, curation, args.words)
    phrases, missing = build_phrases(dictionary, curation)
    if missing:
        print(f"warning: {len(missing)} curated phrase(s) not in Wiktionary: {missing}")
    deck = words + phrases
    surface_rank = {word: rank for rank, word in reversed(ranked)}
    examples = select_examples(
        deck, load_corpus(args.raw_dir), dictionary, surface_rank, curation.meanings
    )

    if args.dry_run:
        for item in deck:
            print(format_item(item, examples.get(item.lemma.key)))
        return 0

    from forglemmigej_api.db import SessionLocal

    with SessionLocal().begin() as session:
        planned = upsert_deck(session, deck, examples)
        deleted, suspended = prune(session, planned)
        counts = dict(
            session.execute(
                select(Card.card_type, func.count()).group_by(Card.card_type)
            ).all()
        )
    with_cards = sum(1 for types in planned.values() if types)
    print(
        f"imported {len(deck)} items ({len(words)} words, {len(phrases)} phrases); "
        f"{len(examples)} with an example sentence, {with_cards} with cards; "
        f"removed {deleted}, suspended {suspended} stale cards; cards by type: "
        + ", ".join(f"{t}={n}" for t, n in sorted(counts.items()))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
