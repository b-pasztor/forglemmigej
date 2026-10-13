"""Deck building on a tiny, hand-written dictionary (no downloaded data needed)."""

from pathlib import Path

import pytest

from forglemmigej_api.content.deck import (
    Curation,
    build_phrases,
    build_words,
    read_frequency_list,
)
from forglemmigej_api.content.wiktionary import Dictionary


def sense(gloss: str, *tags: str, form_of: str | None = None) -> dict:
    s: dict = {"glosses": [gloss], "tags": list(tags)}
    if form_of:
        s["form_of"] = [{"word": form_of}]
        s["tags"].append("form-of")
    return s


ENTRIES = [
    {
        "word": "for",
        "pos": "noun",
        "senses": [sense("lining (inside of a coat)", "neuter")],
    },
    {"word": "for", "pos": "prep", "senses": [sense("for")]},
    {"word": "hus", "pos": "noun", "senses": [sense("house", "neuter")]},
    {
        "word": "hus",
        "pos": "verb",
        "senses": [sense("imperative of huse", "imperative", form_of="huse")],
    },
    {"word": "huse", "pos": "verb", "senses": [sense("to house")]},
    {
        "word": "huset",
        "pos": "noun",
        "senses": [sense("definite of hus", form_of="hus")],
    },
    {"word": "skal", "pos": "noun", "senses": [sense("shell", "common-gender")]},
    {
        "word": "skal",
        "pos": "verb",
        "senses": [sense("present of skulle", "present", form_of="skulle")],
    },
    {"word": "skulle", "pos": "verb", "senses": [sense("shall")]},
    {"word": "tak", "pos": "noun", "senses": [sense("thanks", "common-gender")]},
    {"word": "lort", "pos": "intj", "senses": [sense("crap", "vulgar")]},
    {
        "word": "sygehus",
        "pos": "noun",
        "senses": [sense("hospital", "neuter")],
        "etymology_templates": [
            {"name": "af", "args": {"1": "da", "2": "syg", "3": "-e-", "4": "hus"}}
        ],
    },
    {
        "word": "ingen",
        "pos": "pron",
        "senses": [sense("nobody")],
        "etymology_templates": [
            {"name": "af", "args": {"1": "non", "2": "einn", "3": "-gi"}}
        ],
    },
    {"word": "mange tak", "pos": "intj", "senses": [sense("thank you very much")]},
    {"word": "Danmark", "pos": "name", "senses": [sense("Denmark")]},
]


@pytest.fixture
def dictionary() -> Dictionary:
    return Dictionary(ENTRIES)


def test_function_word_reading_wins(dictionary: Dictionary) -> None:
    assert dictionary.lookup("for").pos == "prep"


def test_inflected_form_maps_to_lemma(dictionary: Dictionary) -> None:
    assert dictionary.lookup("huset").word == "hus"
    assert dictionary.lookup("skal").word == "skulle"


def test_bare_imperative_reading_is_not_preferred(dictionary: Dictionary) -> None:
    lemma = dictionary.lookup("hus")
    assert (lemma.word, lemma.pos, lemma.article) == ("hus", "noun", "et")


def test_names_and_vulgar_entries_are_skipped(dictionary: Dictionary) -> None:
    assert dictionary.lookup("Danmark") is None
    assert dictionary.lookup("lort") is None


def test_only_danish_compound_parts_are_used(dictionary: Dictionary) -> None:
    assert dictionary.lookup("sygehus").compound_parts == ("syg", "-e-", "hus")
    assert dictionary.lookup("ingen").compound_parts == ()


def test_gloss_explanations_are_dropped(dictionary: Dictionary) -> None:
    assert dictionary.lookup("for", prefer_pos="noun").english == "lining"


def test_words_are_distinct_lemmas_in_frequency_order(dictionary: Dictionary) -> None:
    frequency = read_frequency_list(
        ["skal 90", "huset 80", "hus 70", "tak 60", "xyz 50"]
    )
    items = build_words(dictionary, frequency, Curation(), limit=10)
    assert [(i.lemma.word, i.frequency_rank, i.seen_as) for i in items] == [
        ("skulle", 1, "skal"),
        ("hus", 2, "huset"),
        ("tak", 4, None),
    ]


def test_curation_overrides_and_excludes(dictionary: Dictionary) -> None:
    curation = Curation(
        lemma_overrides={"skal": ("skal", "noun")},
        fixes={"tak|noun": {"english": "thank you"}},
        exclude=frozenset({"hus"}),
    )
    frequency = read_frequency_list(["skal 3", "hus 2", "tak 1"])
    items = build_words(dictionary, frequency, curation, limit=10)
    assert [(i.lemma.word, i.lemma.english) for i in items] == [
        ("skal", "shell"),
        ("tak", "thank you"),
    ]


def test_phrases_report_missing_entries(dictionary: Dictionary) -> None:
    curation = Curation(
        phrases={"mange tak": ("politeness",), "god morgen": ("greetings",)}
    )
    items, missing = build_phrases(dictionary, curation)
    assert [(i.lemma.word, i.topics) for i in items] == [("mange tak", ("politeness",))]
    assert missing == ["god morgen"]


def test_repository_curation_files_load() -> None:
    curation_dir = Path(__file__).resolve().parents[2] / "data" / "curation"
    curation = Curation.load(curation_dir)
    assert curation.pos_overrides["så"] == "adv"
    assert curation.lemma_overrides["gør"] == ("gøre", "verb")
    assert "hej" in curation.phrases
