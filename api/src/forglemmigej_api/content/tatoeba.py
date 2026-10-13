"""Read Tatoeba's per-language exports: Danish sentences and their English translations.

Files (from https://downloads.tatoeba.org/exports/per_language/, unpacked):
    dan_sentences.tsv   id <TAB> lang <TAB> text
    dan-eng_links.tsv   danish_id <TAB> english_id
    eng_sentences.tsv   id <TAB> lang <TAB> text
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

TOKEN = re.compile(r"[a-zæøåéü]+(?:'[a-z]+)?", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN.findall(text)]


@dataclass(frozen=True)
class Translation:
    id: str
    text: str


@dataclass(frozen=True)
class Sentence:
    id: str
    text: str
    translations: tuple[Translation, ...]

    @property
    def tokens(self) -> list[str]:
        return tokenize(self.text)


def _read_tsv(path: Path) -> dict[str, str]:
    sentences = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            sentence_id, _lang, text = line.rstrip("\n").split("\t", 2)
            sentences[sentence_id] = text
    return sentences


def load_corpus(raw_dir: Path) -> list[Sentence]:
    """Danish sentences that have at least one English translation."""
    danish = _read_tsv(raw_dir / "dan_sentences.tsv")
    links: dict[str, list[str]] = defaultdict(list)
    with (raw_dir / "dan-eng_links.tsv").open(encoding="utf-8") as f:
        for line in f:
            danish_id, english_id = line.split()
            links[danish_id].append(english_id)
    wanted = {e for ids in links.values() for e in ids}
    english = {}
    with (raw_dir / "eng_sentences.tsv").open(encoding="utf-8") as f:
        for line in f:
            sentence_id, _lang, text = line.rstrip("\n").split("\t", 2)
            if sentence_id in wanted:
                english[sentence_id] = text
    return [
        Sentence(
            id=sentence_id,
            text=text,
            translations=tuple(
                Translation(e, english[e]) for e in links[sentence_id] if e in english
            ),
        )
        for sentence_id, text in danish.items()
        if any(e in english for e in links.get(sentence_id, []))
    ]
