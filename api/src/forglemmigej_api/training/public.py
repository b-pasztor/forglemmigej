"""Public review logs (open-spaced-repetition/fsrs-dataset, MIT) for pre-training.

Each file holds one Anki user's reviews (`v4/revlog_history-<user>.tsv`):
    review_time  epoch milliseconds
    card_id      Anki card id
    review_rating  1..4 (Again, Hard, Good, Easy), the same scale as ours
They have no response times and no card details, so only the history features
can be computed for them.
"""

import csv
from collections import defaultdict
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from forglemmigej_api.features import PastReview
from forglemmigej_api.training.dataset import Example, examples_from_history


def read_user_histories(path: Path) -> dict[str, list[PastReview]]:
    """Card id -> chronological reviews for one user file."""
    user = path.stem.removeprefix("revlog_history-")
    histories: dict[str, list[PastReview]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            reviewed_at = datetime.fromtimestamp(int(row["review_time"]) / 1000, tz=UTC)
            histories[f"{user}:{row['card_id']}"].append(
                PastReview(reviewed_at, int(row["review_rating"]))
            )
    for reviews in histories.values():
        reviews.sort(key=lambda r: r.reviewed_at)
    return histories


def public_examples(path: Path, max_cards: int | None = None) -> Iterator[Example]:
    for i, (card_id, reviews) in enumerate(read_user_histories(path).items()):
        if max_cards is not None and i >= max_cards:
            break
        yield from examples_from_history(card_id, reviews)
