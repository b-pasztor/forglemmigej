"""Build the learning deck: frequent words and curated survival phrases."""

import tomllib
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path

from forglemmigej_api.content.wiktionary import Dictionary, Lemma


@dataclass(frozen=True)
class DeckItem:
    lemma: Lemma
    kind: str  # "word" or "phrase"
    frequency_rank: int | None
    topics: tuple[str, ...] = ()
    seen_as: str | None = None  # the surface form found in the frequency list


@dataclass(frozen=True)
class Curation:
    """Hand-maintained corrections, kept in data/curation/*.toml."""

    # Frequent ambiguous word forms -> the part of speech to use.
    pos_overrides: dict[str, str] = field(default_factory=dict)
    # Word form -> (lemma, part of speech) when the automatic choice is wrong.
    lemma_overrides: dict[str, tuple[str, str]] = field(default_factory=dict)
    # Lemma key ("word|pos") -> replacement fields ("danish", "english").
    fixes: dict[str, dict[str, str]] = field(default_factory=dict)
    # Lemma key -> English words a translation must contain to confirm the meaning.
    meanings: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # Word forms to leave out (e.g. English words common in subtitles).
    exclude: frozenset[str] = frozenset()
    # Phrase -> topics.
    phrases: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def apply_fix(self, lemma: Lemma) -> Lemma:
        fix = self.fixes.get(lemma.key)
        if not fix:
            return lemma
        return replace(
            lemma,
            word=fix.get("danish", lemma.word),
            english=fix.get("english", lemma.english),
        )

    @classmethod
    def load(cls, directory: Path) -> "Curation":
        words = tomllib.loads(
            (directory / "word_overrides.toml").read_text(encoding="utf-8")
        )
        phrases = tomllib.loads(
            (directory / "survival_phrases.toml").read_text(encoding="utf-8")
        )
        return cls(
            pos_overrides=dict(words.get("pos", {})),
            lemma_overrides={
                form: tuple(target.split("|", 1))  # type: ignore[misc]
                for form, target in words.get("lemma", {}).items()
            },
            fixes=dict(words.get("fix", {})),
            meanings={k: tuple(v) for k, v in words.get("meaning", {}).items()},
            exclude=frozenset(words.get("exclude", {}).get("words", [])),
            phrases={
                phrase: (topic,)
                for topic, entries in phrases.items()
                for phrase in entries["phrases"]
            },
        )


def read_frequency_list(lines: Iterable[str]) -> Iterable[tuple[int, str]]:
    """Yield (rank, word) from "word count" lines, most frequent first."""
    for rank, line in enumerate(lines, 1):
        parts = line.split()
        if parts:
            yield rank, parts[0]


def build_words(
    dictionary: Dictionary,
    frequency: Iterable[tuple[int, str]],
    curation: Curation,
    limit: int,
) -> list[DeckItem]:
    """The `limit` most frequent distinct lemmas, in frequency order."""
    items: list[DeckItem] = []
    seen: set[str] = set()
    for rank, word in frequency:
        if word in curation.exclude:
            continue
        target_lemma, target_pos = curation.lemma_overrides.get(word, (None, None))
        lemma = dictionary.lookup(
            word, target_pos or curation.pos_overrides.get(word), target_lemma
        )
        if lemma is None or lemma.key in seen or not lemma.english:
            continue
        seen.add(lemma.key)
        lemma = curation.apply_fix(lemma)
        items.append(
            DeckItem(
                lemma=lemma,
                kind="word",
                frequency_rank=rank,
                seen_as=word if word != lemma.word else None,
            )
        )
        if len(items) >= limit:
            break
    return items


def build_phrases(
    dictionary: Dictionary, curation: Curation
) -> tuple[list[DeckItem], list[str]]:
    """Curated phrases that exist in the dictionary, plus the ones that don't."""
    items, missing = [], []
    for phrase, topics in curation.phrases.items():
        lemma = dictionary.lookup(phrase)
        if lemma is None:
            missing.append(phrase)
            continue
        items.append(
            DeckItem(
                lemma=curation.apply_fix(lemma),
                kind="phrase",
                frequency_rank=None,
                topics=topics,
            )
        )
    return items, missing
