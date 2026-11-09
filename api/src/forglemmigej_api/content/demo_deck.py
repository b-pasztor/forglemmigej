"""Build the fixed deck of the public try-it demo.

The demo is the review app built to run in the browser alone: no server, no
personal data. Its deck is the curated survival phrases, prepared the same way
as the real deck (Wiktionary meanings, validated Tatoeba sentences, the same
card plan), and written to a JSON file that is committed and bundled into the
demo build. Every card carries links to its sources, since the deck is derived
from CC BY-SA (Wiktionary) and CC BY (Tatoeba) material.

Usage (from api/, needs the raw sources in data/raw):
    uv run python -m forglemmigej_api.content.demo_deck
"""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from urllib.parse import quote

from forglemmigej_api.app import first_letters, mask
from forglemmigej_api.content.deck import Curation, DeckItem, build_phrases
from forglemmigej_api.content.examples import Example, select_examples
from forglemmigej_api.content.importer import REPO_ROOT, plan_cards
from forglemmigej_api.content.tatoeba import load_corpus
from forglemmigej_api.content.wiktionary import Dictionary
from forglemmigej_api.models import CardType

OUTPUT = REPO_ROOT / "web" / "src" / "lib" / "demo" / "deck.json"
LICENSE = {
    "name": "CC BY-SA 4.0",
    "url": "https://creativecommons.org/licenses/by-sa/4.0/",
}


def wiktionary_url(word: str) -> str:
    return f"https://en.wiktionary.org/wiki/{quote(word.replace(' ', '_'))}#Danish"


def tatoeba_url(sentence_id: str) -> str:
    return f"https://tatoeba.org/en/sentences/show/{sentence_id}"


def hint(card_type: CardType, phrase: str) -> str | None:
    """First letters of a multi-word phrase on typed and recall cards (as the API)."""
    if card_type in (CardType.CLOZE, CardType.RECALL) and " " in phrase.strip():
        return first_letters(phrase)
    return None


def gap_found(example: Example) -> bool:
    return mask(example.sentence.text, example.target_form) != example.sentence.text


def demo_cards(items: Sequence[DeckItem], examples: dict[str, Example]) -> list[dict]:
    """The cards in curriculum order; a word's second card names its first one."""
    cards: list[dict] = []
    for item in items:
        lemma = item.lemma
        example = examples.get(lemma.key)
        first_id = None
        for plan in plan_cards(item, example):
            if plan.card_type == CardType.CLOZE and not gap_found(plan.sentence):
                # "Nej, tak." cannot hide "nej tak": the answer would be on the card.
                plan = replace(plan, card_type=CardType.RECALL, sentence=None)
            card_id = len(cards) + 1
            is_cloze = plan.card_type == CardType.CLOZE
            target = plan.sentence.target_form if plan.sentence else None
            sources = [{"name": "Wiktionary", "url": wiktionary_url(lemma.word)}]
            context = None
            if plan.sentence:
                sentence, translation = (
                    plan.sentence.sentence,
                    plan.sentence.translation,
                )
                context = {
                    "danish": mask(sentence.text, target)
                    if is_cloze
                    else sentence.text,
                    "english": translation.text,
                    "target_form": None if is_cloze else target,
                }
                sources += [
                    {
                        "name": f"Tatoeba #{sentence.id}",
                        "url": tatoeba_url(sentence.id),
                    },
                    {
                        "name": f"Tatoeba #{translation.id}",
                        "url": tatoeba_url(translation.id),
                    },
                ]
            cards.append(
                {
                    "card_id": card_id,
                    "card_type": plan.card_type.value,
                    "topic": item.topics[0] if item.topics else None,
                    "unlocked_by": first_id,
                    "expected": target or lemma.word,
                    "full_sentence": plan.sentence.sentence.text
                    if plan.sentence
                    else None,
                    "hint": hint(plan.card_type, target or lemma.word),
                    "word": {
                        "danish": lemma.word,
                        "article": lemma.article,
                        "english": lemma.english,
                        "pos": lemma.pos,
                        "glosses": list(lemma.glosses),
                        "ipa": lemma.ipa,
                        "compound_parts": list(lemma.compound_parts),
                    },
                    "context": context,
                    "sources": sources,
                }
            )
            first_id = first_id or card_id
    return cards


def build(raw_dir: Path, curation_dir: Path) -> dict:
    dictionary = Dictionary.from_jsonl(raw_dir / "kaikki-danish.jsonl")
    curation = Curation.load(curation_dir)
    phrases, missing = build_phrases(dictionary, curation)
    if missing:
        print(f"warning: {len(missing)} curated phrase(s) not in Wiktionary: {missing}")
    examples = select_examples(
        phrases, load_corpus(raw_dir), dictionary, {}, curation.meanings
    )
    return {
        "license": LICENSE,
        "attribution": [
            {
                "name": "Wiktionary (via kaikki.org)",
                "url": "https://en.wiktionary.org/",
                "license": "CC BY-SA 4.0",
            },
            {
                "name": "Tatoeba",
                "url": "https://tatoeba.org/",
                "license": "CC BY 2.0 FR",
            },
        ],
        "cards": demo_cards(phrases, examples),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument(
        "--curation-dir", type=Path, default=REPO_ROOT / "data" / "curation"
    )
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    deck = build(args.raw_dir, args.curation_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as f:  # LF on Windows too
        f.write(json.dumps(deck, ensure_ascii=False, indent=1) + "\n")
    types = [c["card_type"] for c in deck["cards"]]
    print(
        f"wrote {args.out}: {len(types)} cards ("
        + ", ".join(f"{t}={types.count(t)}" for t in sorted(set(types)))
        + ")"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
