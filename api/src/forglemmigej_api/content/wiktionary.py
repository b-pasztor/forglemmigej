"""Read the Danish Wiktionary extract (kaikki.org JSONL) and map words to lemmas.

A frequency list counts every surface form separately ("er", "var" and "været"
are all forms of "være"). This module maps such forms to their dictionary entry
and picks the most likely reading for frequent words, which are often ambiguous
("for" is a preposition, but also a noun meaning "lining").
"""

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Entry = Mapping[str, Any]

# Entry types that are never learning material.
SKIP_POS = frozenset(
    {
        "name",
        "character",
        "symbol",
        "punct",
        "prefix",
        "suffix",
        "interfix",
        "circumfix",
        "circumpos",
    }
)

# Closed word classes first: for very frequent words the function-word reading is
# almost always the intended one.
POS_PRIORITY = (
    "pron", "conj", "prep", "article", "det", "intj", "num", "particle",
    "adv", "verb", "noun", "adj", "phrase", "contraction",
)  # fmt: skip

# Senses with these tags are not suitable for a beginner deck.
UNSUITABLE_TAGS = frozenset(
    {"vulgar", "offensive", "derogatory", "slur", "obsolete", "archaic"}
)

# A bare imperative ("hus!" = "house [someone]!") is rarely the reading behind a
# frequent word form; prefer other readings.
IMPERATIVE_PENALTY = 50


def _pos_rank(pos: str) -> int:
    return POS_PRIORITY.index(pos) if pos in POS_PRIORITY else len(POS_PRIORITY)


def _tags(entry: Entry) -> set[str]:
    return {t for s in entry.get("senses", []) for t in (s.get("tags") or [])}


def is_form_of(entry: Entry) -> bool:
    """True if every sense only points to another entry (an inflected or alternative form)."""
    senses = entry.get("senses", [])
    return bool(senses) and all(
        {"form-of", "alt-of"} & set(s.get("tags") or []) for s in senses
    )


def is_unsuitable(entry: Entry) -> bool:
    senses = entry.get("senses", [])
    return bool(senses) and all(
        UNSUITABLE_TAGS & set(s.get("tags") or []) for s in senses
    )


@dataclass(frozen=True)
class Lemma:
    """A dictionary entry prepared for the learning deck."""

    word: str
    pos: str
    english: str
    article: str | None = None
    glosses: tuple[str, ...] = ()
    ipa: str | None = None
    compound_parts: tuple[str, ...] = ()
    tags: tuple[str, ...] = field(default=())

    @property
    def key(self) -> str:
        return f"{self.word}|{self.pos}"


def _clean_gloss(gloss: str) -> str:
    """Drop parenthesised explanations; they make card backs long."""
    gloss = re.sub(r"\s*\([^)]*\)", "", gloss)
    return gloss.replace(")", "").replace("(", "").strip()


def _compound_parts(entry: Entry) -> tuple[str, ...]:
    for template in entry.get("etymology_templates", []):
        args = template.get("args", {})
        # Only Danish word formation; etymologies also describe Old Norse ancestors.
        if (
            template.get("name") in ("af", "affix", "compound")
            and args.get("1") == "da"
        ):
            numbered = sorted((k for k in args if k.isdigit() and k != "1"), key=int)
            parts = tuple(args[k] for k in numbered if args[k])
            if len(parts) >= 2:
                return parts
    return ()


def to_lemma(entry: Entry) -> Lemma:
    senses = [s for s in entry.get("senses", []) if s.get("glosses")]
    glosses = tuple(dict.fromkeys(_clean_gloss(s["glosses"][0]) for s in senses))
    tags = _tags(entry)
    article = None
    if entry["pos"] == "noun":
        article = (
            "et" if "neuter" in tags else "en" if "common-gender" in tags else None
        )
    ipa = next((s["ipa"] for s in entry.get("sounds", []) if s.get("ipa")), None)
    return Lemma(
        word=entry["word"],
        pos=entry["pos"],
        english=" / ".join(glosses[:2]),
        article=article,
        glosses=glosses,
        ipa=ipa,
        compound_parts=_compound_parts(entry),
        tags=tuple(sorted(tags)),
    )


class Dictionary:
    def __init__(self, entries: Iterable[Entry]) -> None:
        self._by_word: dict[str, list[Entry]] = defaultdict(list)
        for entry in entries:
            if entry.get("pos") not in SKIP_POS:
                self._by_word[entry["word"]].append(entry)

    @classmethod
    def from_jsonl(cls, path: Path) -> "Dictionary":
        with path.open(encoding="utf-8") as f:
            return cls(json.loads(line) for line in f)

    def entries(self, word: str) -> list[Entry]:
        return self._by_word.get(word, [])

    def _targets(self, entry: Entry) -> Iterable[Entry]:
        """Dictionary entries an inflected or alternative form points to."""
        for sense in entry.get("senses", []):
            for ref in sense.get("form_of", []) + sense.get("alt_of", []):
                for target in self.entries(ref["word"]):
                    if target["pos"] == entry["pos"] and not is_form_of(target):
                        yield target

    def candidates(
        self, word: str, prefer_pos: str | None = None, prefer_lemma: str | None = None
    ) -> list[Entry]:
        """Possible dictionary entries behind a word form, most likely first."""
        scored: list[tuple[int, int, Entry]] = []
        for order, entry in enumerate(self.entries(word)):
            if not is_form_of(entry):
                scored.append((_pos_rank(entry["pos"]), order, entry))
                continue
            penalty = IMPERATIVE_PENALTY if "imperative" in _tags(entry) else 0
            for target in self._targets(entry):
                scored.append((_pos_rank(target["pos"]) + penalty, order, target))
        scored = [c for c in scored if not is_unsuitable(c[2])]
        if prefer_lemma:
            scored = [c for c in scored if c[2]["word"] == prefer_lemma] or scored
        if prefer_pos:
            scored = [c for c in scored if c[2]["pos"] == prefer_pos] or scored
        scored.sort(key=lambda c: (c[0], c[1]))
        return [entry for _, _, entry in scored]

    def lookup(
        self, word: str, prefer_pos: str | None = None, prefer_lemma: str | None = None
    ) -> Lemma | None:
        found = self.candidates(word, prefer_pos, prefer_lemma)
        return to_lemma(found[0]) if found else None
