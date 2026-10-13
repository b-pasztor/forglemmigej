"""Import the learning deck into the database.

Usage (from api/):
    uv run python -m forglemmigej_api.content.importer --words 500
    uv run python -m forglemmigej_api.content.importer --words 300 --dry-run

The import is idempotent: items are upserted by (source, source_ref) and each
item gets a recognition and a recall card if it does not have them yet.
Items that are no longer in the deck (e.g. after a curation fix) are deleted,
or suspended if they have already been reviewed, so the review log stays intact.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy import delete, func, literal_column, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from forglemmigej_api.content.deck import (
    Curation,
    DeckItem,
    build_phrases,
    build_words,
    read_frequency_list,
)
from forglemmigej_api.content.wiktionary import Dictionary
from forglemmigej_api.models import Card, CardStatus, CardType, Item, Review

REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE = "wiktionary"
NEW_ITEM_CARD_TYPES = (CardType.RECOGNITION, CardType.RECALL)


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


def upsert_items(session: Session, items: Sequence[DeckItem]) -> int:
    """Insert or update items and make sure each has its cards. Returns the item count."""
    for item in items:
        stmt = insert(Item).values(**item_values(item))
        excluded = stmt.excluded
        stmt = stmt.on_conflict_do_update(
            index_elements=["source", "source_ref"],
            set_={
                "kind": excluded.kind,
                "danish": excluded.danish,
                "english": excluded.english,
                "article": excluded.article,
                "details": excluded.details,
                # A phrase can also be a frequent word ("hej"): keep the rank, merge topics.
                "frequency_rank": func.coalesce(
                    excluded.frequency_rank, Item.frequency_rank
                ),
                "topics": func.coalesce(
                    select(func.array_agg(literal_column("t").distinct()))
                    .select_from(
                        func.unnest(Item.topics.concat(excluded.topics)).alias("t")
                    )
                    .scalar_subquery(),
                    literal_column("'{}'::text[]"),
                ),
            },
        ).returning(Item.id)
        item_id = session.execute(stmt).scalar_one()
        session.execute(
            insert(Card)
            .values([{"item_id": item_id, "card_type": t} for t in NEW_ITEM_CARD_TYPES])
            .on_conflict_do_nothing(index_elements=["item_id", "card_type"])
        )
    return len(items)


def prune_items(session: Session, keep: Sequence[DeckItem]) -> tuple[int, int]:
    """Remove items that left the deck. Returns (deleted, suspended) item counts."""
    keep_refs = [item.lemma.key for item in keep]
    stale = select(Item.id).where(
        Item.source == SOURCE, Item.source_ref.not_in(keep_refs)
    )
    reviewed = select(Card.item_id).join(Review, Review.card_id == Card.id).distinct()
    suspended = session.execute(
        update(Card)
        .where(Card.item_id.in_(stale), Card.item_id.in_(reviewed))
        .values(status=CardStatus.SUSPENDED)
        .returning(Card.item_id)
    ).scalars()
    unreviewed = stale.where(Item.id.not_in(reviewed))
    deletable = session.scalars(unreviewed).all()
    session.execute(delete(Card).where(Card.item_id.in_(deletable)))
    session.execute(delete(Item).where(Item.id.in_(deletable)))
    return len(deletable), len(set(suspended))


def format_item(item: DeckItem) -> str:
    lemma = item.lemma
    head = f"{lemma.article} {lemma.word}" if lemma.article else lemma.word
    rank = f"#{item.frequency_rank}" if item.frequency_rank else "phrase"
    extra = []
    if item.seen_as:
        extra.append(f"seen as '{item.seen_as}'")
    if lemma.compound_parts:
        extra.append(" + ".join(lemma.compound_parts))
    if item.topics:
        extra.append(",".join(item.topics))
    suffix = f"  ({'; '.join(extra)})" if extra else ""
    return f"{rank:>7}  {head:<22} {lemma.pos:<7} {lemma.english[:60]}{suffix}"


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
        words = build_words(dictionary, read_frequency_list(f), curation, args.words)
    phrases, missing = build_phrases(dictionary, curation)
    if missing:
        print(f"warning: {len(missing)} curated phrase(s) not in Wiktionary: {missing}")

    if args.dry_run:
        for item in words + phrases:
            print(format_item(item))
        return 0

    from forglemmigej_api.db import SessionLocal

    with SessionLocal.begin() as session:
        count = upsert_items(session, words + phrases)
        deleted, suspended = prune_items(session, words + phrases)
        cards = session.scalar(select(func.count()).select_from(Card))
    print(
        f"imported {count} items ({len(words)} words, {len(phrases)} phrases); "
        f"removed {deleted}, suspended {suspended} stale items; {cards} cards in total"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
