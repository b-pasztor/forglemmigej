"""Example-sentence selection on hand-written mini data."""

import pytest

from forglemmigej_api.content.deck import DeckItem
from forglemmigej_api.content.examples import (
    confirmed_meaning,
    meaning_keys,
    select_examples,
)
from forglemmigej_api.content.tatoeba import Sentence, Translation
from forglemmigej_api.content.wiktionary import Dictionary


def sense(gloss: str, *tags: str, form_of: str | None = None) -> dict:
    s: dict = {"glosses": [gloss], "tags": list(tags)}
    if form_of:
        s["form_of"] = [{"word": form_of}]
        s["tags"].append("form-of")
    return s


DICTIONARY = Dictionary(
    [
        {"word": "ved", "pos": "prep", "senses": [sense("at, by")]},
        {
            "word": "ved",
            "pos": "verb",
            "senses": [sense("present of vide", form_of="vide")],
        },
        {
            "word": "vide",
            "pos": "verb",
            "senses": [sense("to know (be sure about (something))")],
        },
        {
            "word": "hus",
            "pos": "noun",
            "senses": [sense("house", "neuter")],
            "forms": [{"form": "huset", "tags": ["definite"]}],
        },
        {"word": "ham", "pos": "pron", "senses": [sense("him: objective of han")]},
        {
            "word": "han",
            "pos": "pron",
            "senses": [sense("he")],
            "forms": [{"form": "ham"}],
        },
        {"word": "tak", "pos": "intj", "senses": [sense("thank you, thanks")]},
        {"word": "takke", "pos": "verb", "senses": [sense("to thank")]},
        {"word": "hvad så", "pos": "phrase", "senses": [sense("what's up")]},
    ]
)


def sentence(sid: str, danish: str, *english: str) -> Sentence:
    return Sentence(
        sid, danish, tuple(Translation(f"{sid}e{i}", e) for i, e in enumerate(english))
    )


def item(word: str, pos: str, rank: int | None = 1) -> DeckItem:
    lemma = DICTIONARY.lookup(word, pos)
    assert lemma is not None
    return DeckItem(lemma=lemma, kind="word", frequency_rank=rank)


def test_meaning_keys_handle_colons_and_nested_parentheses() -> None:
    assert meaning_keys(["him: objective of han"]) >= {("him",)}
    assert ("know",) in meaning_keys(["to know (be sure about (something))"])


def test_confirmed_meaning_uses_light_stemming() -> None:
    assert confirmed_meaning({("close",)}, "She closed the door.") == "close"
    assert confirmed_meaning({("on",)}, "He is gone.") is None


def test_translation_must_confirm_the_meaning() -> None:
    corpus = [sentence("1", "Huset er stort.", "The house is big.")]
    examples = select_examples([item("hus", "noun")], corpus, DICTIONARY, {})
    example = examples["hus|noun"]
    assert (example.target_form, example.matched_meaning) == ("Huset", "house")

    wrong = [sentence("2", "Huset er stort.", "The building is big.")]
    assert select_examples([item("hus", "noun")], wrong, DICTIONARY, {}) == {}


def test_sentence_confirming_a_competing_reading_is_rejected() -> None:
    corpus = [
        sentence(
            "1", "Ved du det?", "Do you know it, by the way?"
        ),  # "vide", not "ved"
        sentence("2", "Han sidder ved bordet.", "He sits at the table."),
    ]
    examples = select_examples([item("ved", "prep")], corpus, DICTIONARY, {})
    assert examples["ved|prep"].sentence.id == "2"


def test_related_words_are_not_competitors() -> None:
    corpus = [sentence("1", "Tak for i dag.", "Thank you for today.")]
    assert "tak|intj" in select_examples([item("tak", "intj")], corpus, DICTIONARY, {})


def test_pronouns_only_match_their_exact_form() -> None:
    # "han" lists "ham" among its forms; a "ham" sentence must not teach "han".
    corpus = [sentence("1", "Jeg ser ham.", "I see him.")]
    examples = select_examples([item("han", "pron")], corpus, DICTIONARY, {})
    assert examples == {}


def test_phrase_needs_confirmation_or_to_be_the_whole_sentence() -> None:
    phrase = DeckItem(
        lemma=DICTIONARY.lookup("hvad så"), kind="phrase", frequency_rank=None
    )
    corpus = [
        sentence("1", "Hvad så du?", "What did you see?"),
        sentence("2", "Hvad så?", "So what?"),
    ]
    examples = select_examples([phrase], corpus, DICTIONARY, {})
    assert examples["hvad så|phrase"].sentence.id == "2"


def test_easier_sentences_win_and_each_sentence_is_used_once() -> None:
    rank = {"er": 3, "stort": 900, "lille": 50}
    corpus = [
        sentence("1", "Huset er stort.", "The house is big."),
        sentence("2", "Huset er lille.", "The house is small."),
    ]
    examples = select_examples([item("hus", "noun")], corpus, DICTIONARY, rank)
    assert examples["hus|noun"].sentence.id == "2"


@pytest.mark.parametrize("text", ["Det koster 5 kroner.", "Huset."])
def test_unusable_sentences_are_skipped(text: str) -> None:
    corpus = [sentence("1", text, "The house costs 5 crowns.")]
    assert select_examples([item("hus", "noun")], corpus, DICTIONARY, {}) == {}
