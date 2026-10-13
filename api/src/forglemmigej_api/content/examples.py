"""Pick one validated example sentence for each word or phrase in the deck.

A learner cannot tell whether a card is wrong, so every example is checked
automatically: the English translation (written by a person on Tatoeba) must
contain one of the word's dictionary meanings. Sentences that fail the check are
never used. Among valid sentences we prefer short ones whose other words are
frequent, so each card adds roughly one new thing (i+1).
"""

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from forglemmigej_api.content.deck import DeckItem
from forglemmigej_api.content.tatoeba import Sentence, Translation, tokenize
from forglemmigej_api.content.wiktionary import Dictionary, Lemma

MIN_WORDS, MAX_WORDS = 2, 8
UNKNOWN_RANK = 100_000
# Inflection tables of closed-class words list other words (jeg -> mig, min),
# so only the exact form counts for them.
EXACT_FORM_POS = frozenset(
    {"pron", "det", "article", "conj", "prep", "particle", "intj"}
)
GLOSS_STOPWORDS = frozenset(
    {
        "to",
        "a",
        "an",
        "the",
        "be",
        "of",
        "one's",
        "oneself",
        "someone",
        "something",
        "etc",
    }
)


@dataclass(frozen=True)
class Example:
    sentence: Sentence
    translation: Translation
    target_form: str  # the form as written in the sentence ("er" for "være")
    matched_meaning: str | None  # the gloss confirmed by the translation


def target_forms(
    lemma: Lemma, dictionary: Dictionary, seen_as: str | None = None
) -> set[str]:
    forms = {lemma.word.lower()}
    if seen_as:
        forms.add(seen_as.lower())
    if lemma.pos not in EXACT_FORM_POS:
        for entry in dictionary.entries(lemma.word):
            if entry["pos"] == lemma.pos:
                forms |= {
                    f["form"].lower()
                    for f in entry.get("forms", [])
                    if " " not in f["form"]
                    and tokenize(f["form"]) == [f["form"].lower()]
                }
    return forms


def _strip_parentheses(text: str) -> str:
    while True:  # innermost first: "know (be sure about (something))"
        stripped = re.sub(r"\([^()]*\)", "", text)
        if stripped == text:
            return text
        text = stripped


def meaning_keys(glosses: Iterable[str], max_words: int = 2) -> set[tuple[str, ...]]:
    """Short English keys that a translation should contain."""
    keys = set()
    for gloss in glosses:
        # "him: objective of han" -> "him", "objective of han"
        for alternative in re.split(r"[,;/:]", _strip_parentheses(gloss)):
            words = tuple(w for w in tokenize(alternative) if w not in GLOSS_STOPWORDS)
            if 1 <= len(words) <= max_words:
                keys.add(words)
    return keys


def _token_matches(key: str, token: str) -> bool:
    if token == key:
        return True
    # Light stemming: "eat" ~ "eats"/"eating", "close" ~ "closed".
    stem = key.removesuffix("e")
    return len(key) >= 4 and token.startswith(stem)


def confirmed_meaning(keys: Iterable[tuple[str, ...]], english: str) -> str | None:
    tokens = tokenize(english)
    for key in sorted(keys, key=len, reverse=True):
        for i in range(len(tokens) - len(key) + 1):
            if all(_token_matches(k, t) for k, t in zip(key, tokens[i:], strict=False)):
                return " ".join(key)
    return None


def competing_keys(
    form: str, lemma: Lemma, dictionary: Dictionary, own: set[tuple[str, ...]]
) -> set[tuple[str, ...]]:
    """Meanings of other words written the same way ("ved" = "at, by", but also
    "knows", from "vide"). A translation confirming one of these is ambiguous."""
    keys: set[tuple[str, ...]] = set()
    for entry in dictionary.candidates(form):
        if (entry["word"], entry["pos"]) != (lemma.word, lemma.pos):
            glosses = [
                s["glosses"][0] for s in entry.get("senses", []) if s.get("glosses")
            ]
            keys |= meaning_keys(glosses)
    # Related words share meanings ("tak" = thanks, "takke" = to thank): a
    # competing key that overlaps one of our own keys is not a different reading.
    own_tokens = {t for key in own for t in key}
    return {
        key
        for key in keys - own
        if not any(
            _token_matches(o, t) or _token_matches(t, o)
            for t in key
            for o in own_tokens
        )
    }


def _difficulty(
    tokens: Sequence[str], skip: Sequence[str], rank: Mapping[str, int]
) -> tuple:
    others = [rank.get(t, UNKNOWN_RANK) for t in tokens if t not in skip]
    return (max(others, default=0), len(tokens))


def _find_phrase(tokens: list[str], phrase: list[str]) -> bool:
    return any(
        tokens[i : i + len(phrase)] == phrase
        for i in range(len(tokens) - len(phrase) + 1)
    )


def _surface_form(text: str, form: str) -> str:
    """The form with its original capitalisation in the sentence."""
    match = re.search(rf"(?<![\wæøå]){re.escape(form)}(?![\wæøå])", text, re.IGNORECASE)
    return match.group(0) if match else form


def _phrase_candidates(
    lemma: Lemma, by_token: Mapping[str, list[Sentence]], used: set[str]
) -> Iterable[tuple[Sentence, Translation, str | None]]:
    """A phrase counts if the translation confirms its meaning, or if the sentence
    is just the phrase itself ("Mange tak."). "Hvad så du?" is not "hvad så"."""
    phrase = tokenize(lemma.word)
    keys = meaning_keys(lemma.glosses, max_words=4)  # "how are you"
    for sentence in by_token.get(phrase[0], []):
        if sentence.id in used or not _find_phrase(sentence.tokens, phrase):
            continue
        for translation in sentence.translations:
            meaning = confirmed_meaning(keys, translation.text)
            if meaning or sentence.tokens == phrase:
                yield sentence, translation, meaning
                break


def _word_candidates(
    item: DeckItem,
    dictionary: Dictionary,
    by_token: Mapping[str, list[Sentence]],
    used: set[str],
    meaning_overrides: Mapping[str, Sequence[str]],
) -> Iterable[tuple[Sentence, Translation, str, str]]:
    lemma = item.lemma
    if lemma.key in meaning_overrides:
        keys = {tuple(tokenize(k)) for k in meaning_overrides[lemma.key]}
    else:
        keys = meaning_keys(lemma.glosses)
    for form in target_forms(lemma, dictionary, item.seen_as):
        competitors = competing_keys(form, lemma, dictionary, keys)
        for sentence in by_token.get(form, []):
            if sentence.id in used:
                continue
            for translation in sentence.translations:
                meaning = confirmed_meaning(keys, translation.text)
                if meaning and not confirmed_meaning(competitors, translation.text):
                    yield sentence, translation, form, meaning
                    break


def select_examples(
    items: Sequence[DeckItem],
    corpus: Iterable[Sentence],
    dictionary: Dictionary,
    surface_rank: Mapping[str, int],
    meaning_overrides: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Example]:
    """Lemma key -> the best validated example. Items are handled in deck order and
    each sentence is used at most once, so frequent words get the easiest sentences.

    `meaning_overrides` replaces the English keys checked for a lemma (e.g. "være"
    -> be/is/are/was, since English inflects "be" irregularly)."""
    meaning_overrides = meaning_overrides or {}
    by_token: dict[str, list[Sentence]] = {}
    for sentence in corpus:
        if MIN_WORDS <= len(sentence.tokens) <= MAX_WORDS and not re.search(
            r"\d", sentence.text
        ):
            for token in set(sentence.tokens):
                by_token.setdefault(token, []).append(sentence)

    used: set[str] = set()
    chosen: dict[str, Example] = {}
    for item in items:
        lemma = item.lemma
        candidates: list[tuple[tuple, Example]] = []
        if " " in lemma.word:
            phrase = tokenize(lemma.word)
            for sentence, translation, meaning in _phrase_candidates(
                lemma, by_token, used
            ):
                example = Example(
                    sentence,
                    translation,
                    _surface_form(sentence.text, lemma.word),
                    meaning,
                )
                candidates.append(
                    (_difficulty(sentence.tokens, phrase, surface_rank), example)
                )
        else:
            for sentence, translation, form, meaning in _word_candidates(
                item, dictionary, by_token, used, meaning_overrides
            ):
                example = Example(
                    sentence, translation, _surface_form(sentence.text, form), meaning
                )
                candidates.append(
                    (_difficulty(sentence.tokens, [form], surface_rank), example)
                )
        if candidates:
            best = min(candidates, key=lambda c: (c[0], c[1].sentence.id))[1]
            used.add(best.sentence.id)
            chosen[lemma.key] = best
    return chosen
