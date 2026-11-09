from forglemmigej_api.content.deck import DeckItem
from forglemmigej_api.content.demo_deck import demo_cards
from forglemmigej_api.content.examples import Example
from forglemmigej_api.content.tatoeba import Sentence, Translation
from forglemmigej_api.content.wiktionary import Lemma


def phrase(word: str, english: str) -> DeckItem:
    return DeckItem(
        lemma=Lemma(word=word, pos="phrase", english=english, glosses=(english,)),
        kind="phrase",
        frequency_rank=None,
        topics=("politeness",),
    )


def example(text: str, english: str, form: str) -> Example:
    translation = Translation(id="2", text=english)
    return Example(Sentence("1", text, (translation,)), translation, form, english)


def test_cards_follow_the_plan_with_hints_and_sources() -> None:
    item = phrase("mange tak", "thanks a lot")
    recognition, cloze = demo_cards(
        [item], {item.lemma.key: example("Mange tak!", "Thanks a lot!", "Mange tak")}
    )
    assert recognition["context"] == {
        "danish": "Mange tak!",
        "english": "Thanks a lot!",
        "target_form": "Mange tak",
    }
    assert cloze["context"]["danish"] == "___!"
    assert cloze["context"]["target_form"] is None  # the answer is not on the card
    assert (cloze["expected"], cloze["hint"]) == ("Mange tak", "M____ t__")
    assert cloze["unlocked_by"] == recognition["card_id"]
    assert [s["name"] for s in cloze["sources"]] == [
        "Wiktionary",
        "Tatoeba #1",
        "Tatoeba #2",
    ]
    assert cloze["sources"][0]["url"].endswith("/wiki/mange_tak#Danish")


def test_a_gap_that_cannot_be_masked_becomes_a_recall_card() -> None:
    item = phrase("nej tak", "no thank you")
    _, second = demo_cards(
        [item], {item.lemma.key: example("Nej, tak.", "No, thank you.", "nej tak")}
    )
    assert second["card_type"] == "recall"
    assert second["context"] is None
    assert (second["expected"], second["hint"]) == ("nej tak", "n__ t__")
    assert [s["name"] for s in second["sources"]] == ["Wiktionary"]
