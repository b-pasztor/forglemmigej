import base64
import json
from datetime import UTC, datetime, timedelta
from unittest import mock

import pytest
from sqlalchemy.orm import Session

pytest.importorskip("mlflow")  # the ml dependency group

from forglemmigej_api.features import FSRS6_DEFAULT_PARAMETERS
from forglemmigej_api.models import (
    AnswerMode,
    Card,
    CardType,
    Item,
    ItemKind,
    MemoryModel,
    ModelRole,
    Prediction,
    Review,
)
from forglemmigej_api.training import report

NOW = datetime(2026, 11, 8, 17, 30, tzinfo=UTC)  # a Sunday
CHAMPION = "fsrs6-default"


def add_challenger(db: Session) -> None:
    db.add(
        MemoryModel(
            version="fitted",
            kind="fsrs",
            role=ModelRole.CHALLENGER,
            parameters=list(FSRS6_DEFAULT_PARAMETERS),
        )
    )
    db.flush()


def add_cards(db: Session, n: int, second_review: datetime) -> None:
    """n cards, each learned two days before `second_review` and reviewed again
    then, with predictions from both models on the second (usable) review."""
    for i in range(n):
        item = Item(
            kind=ItemKind.WORD,
            danish=f"ord{i}",
            english=f"word {i}",
            source="t",
            source_ref=f"ord{i}|{second_review:%j}",
        )
        db.add(item)
        db.flush()
        card = Card(item_id=item.id, card_type=CardType.RECOGNITION)
        db.add(card)
        db.flush()
        for at, grade in (
            (second_review - timedelta(days=2), 3),
            (second_review, 1 + 2 * (i % 4 != 0)),
        ):
            review = Review(
                card_id=card.id,
                reviewed_at=at,
                grade=grade,
                answer_mode=AnswerMode.SELF_GRADE,
            )
            db.add(review)
            db.flush()
            if at == second_review:
                for version, value in ((CHAMPION, 0.9), ("fitted", 0.75)):
                    db.add(
                        Prediction(
                            review_id=review.id,
                            model_version=version,
                            predicted_recall=value,
                            is_champion=version == CHAMPION,
                        )
                    )
    db.flush()


def test_an_empty_log_gives_an_empty_report(db: Session) -> None:
    built = report.build_report(db, NOW)
    assert built["weeks"] == []
    assert built["totals"] == {"usable_reviews": 0, "first_week": None}
    assert [m["role"] for m in built["models"]] == ["champion"]


def test_weeks_with_few_reviews_show_only_their_count(db: Session) -> None:
    add_challenger(db)
    add_cards(db, 24, NOW - timedelta(days=7))  # Sunday of the previous week
    add_cards(db, 5, NOW)
    built = report.build_report(db, NOW)

    busy, quiet = built["weeks"]
    assert (busy["week"], busy["usable_reviews"]) == ("2026-10-26", 24)
    assert busy["recall_rate"] == 0.75
    assert busy["champion_predicted"] == 0.9
    assert set(busy["models"]) == {CHAMPION, "fitted"}
    assert busy["models"]["fitted"]["n"] == 24
    assert quiet == {"week": "2026-11-02", "usable_reviews": 5}
    assert built["totals"] == {"usable_reviews": 29, "first_week": "2026-10-26"}

    assert set(built["calibration"]) == {CHAMPION, "fitted"}
    [gate] = built["gate"]
    assert gate["challenger"] == "fitted" and not gate["promote"]
    assert any("< 300" in reason for reason in gate["reasons"])


def test_the_report_holds_no_review_level_data(db: Session) -> None:
    add_challenger(db)
    add_cards(db, 24, NOW - timedelta(days=3))
    text = report.render(report.build_report(db, NOW)).decode()
    for secret in ("ord1", "word 1", "card_id", "review_id", "T17:30", "T16:30"):
        assert secret not in text


def test_publish_commits_only_when_the_content_changed() -> None:
    content = b'{"schema": 1}\n'
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request.get_method(), request.full_url))
        body = {"content": base64.b64encode(content).decode(), "sha": "abc"}
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps(body).encode()
        return response

    with mock.patch("urllib.request.urlopen", fake_urlopen):
        assert not report.publish(content, "o/r", "report-data", "report.json", "t")
        assert report.publish(b"{}\n", "o/r", "report-data", "report.json", "t")
    assert [method for method, _ in calls] == ["GET", "GET", "PUT"]
    assert calls[0][1].endswith("/repos/o/r/contents/report.json?ref=report-data")


def test_publishing_without_a_token_is_skipped(monkeypatch) -> None:
    monkeypatch.delenv("REPORT_GITHUB_TOKEN", raising=False)
    assert report.publish_from_environment({}).startswith("skipped")
