"""The public model report: weekly aggregates of how well each model predicts.

Published as static JSON (rendered by `report/index.html` on GitHub Pages), so
it stays online while the homelab is off. Only aggregates leave the homelab:
no cards, words, answers or single reviews, and no daily activity. Weeks with
too few reviews show only their count, since a metric on a handful of reviews
is noise (and closer to an individual review).

Usage (writes the JSON; with --publish also commits it to the data branch):
    uv run --group ml python -m forglemmigej_api.training.report --out report.json
"""

import argparse
import base64
import json
import os
import urllib.error
import urllib.request
from collections.abc import Sequence
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from forglemmigej_api.models import MemoryModel, ModelPromotion, ModelRole
from forglemmigej_api.training import gate
from forglemmigej_api.training.evaluate import MIN_GAP_DAYS, calibration_table
from forglemmigej_api.training.fit_fsrs import DEFAULT_MIN_REVIEWS
from forglemmigej_api.training.nightly import evaluate_models, shadow_predictions

TIMEZONE = ZoneInfo("Europe/Zurich")
MIN_WEEK_REVIEWS = 20  # below this a week shows only its review count
CALIBRATION_BINS = 10
SCHEMA_VERSION = 1


def _week_start(timestamps: pd.Series) -> pd.Series:
    local = timestamps.dt.tz_convert(TIMEZONE)
    return (local.dt.normalize() - pd.to_timedelta(local.dt.weekday, unit="D")).dt.date


def _rounded(value: float, digits: int = 4) -> float | None:
    return None if pd.isna(value) else round(float(value), digits)


def weekly(predictions: pd.DataFrame) -> list[dict]:
    """Per week (Monday, Zurich time): usable reviews, actual recall vs the
    champion's prediction (the drift view), and each model's scores on the
    reviews that every model predicting that week predicted."""
    usable = predictions[predictions["gap_days"] >= MIN_GAP_DAYS].copy()
    if usable.empty:
        return []
    usable["week"] = _week_start(usable["reviewed_at"])
    weeks = []
    for week, part in usable.groupby("week"):
        champion = part[part["is_champion"]]
        n = int(champion["review_id"].nunique())
        entry: dict = {"week": week.isoformat(), "usable_reviews": n}
        if n >= MIN_WEEK_REVIEWS:
            scores = evaluate_models(part)
            entry |= {
                "recall_rate": _rounded(champion["recalled"].mean()),
                "champion_predicted": _rounded(champion["predicted_recall"].mean()),
                "models": {
                    version: {
                        "log_loss": _rounded(row["log_loss"]),
                        "rmse_bins": _rounded(row["rmse_bins"]),
                        "n": int(row["n"]),
                    }
                    for version, row in scores.iterrows()
                },
            }
        weeks.append(entry)
    return weeks


def calibration(predictions: pd.DataFrame, versions: Sequence[str]) -> dict:
    """All-time calibration of the active models, on the reviews they all
    predicted: per probability bin, the mean prediction and the actual rate."""
    usable = predictions[
        (predictions["gap_days"] >= MIN_GAP_DAYS)
        & predictions["model_version"].isin(versions)
    ]
    if usable.empty:
        return {}
    n_models = usable["model_version"].nunique()
    shared = usable[
        usable.groupby("review_id")["model_version"].transform("nunique") == n_models
    ]
    result = {}
    for version, part in shared.groupby("model_version"):
        table = calibration_table(
            part["recalled"], part["predicted_recall"], CALIBRATION_BINS
        )
        result[version] = [
            {
                "predicted": _rounded(row.predicted, 3),
                "actual": _rounded(row.actual, 3),
                "n": int(row.n),
            }
            for row in table.itertuples()
        ]
    return result


def weekly_first(usable: pd.DataFrame) -> str | None:
    if usable.empty:
        return None
    return min(_week_start(usable["reviewed_at"])).isoformat()


def build_report(session: Session, now: datetime) -> dict:
    predictions = shadow_predictions(session, datetime(1970, 1, 1, tzinfo=UTC))
    models = session.scalars(select(MemoryModel).order_by(MemoryModel.created_at)).all()
    active = [m.version for m in models if m.role != ModelRole.RETIRED]
    decisions, _ = gate.run_gate(session, predictions, apply=False)
    promotions = session.scalars(
        select(ModelPromotion).order_by(ModelPromotion.promoted_at)
    ).all()
    usable = predictions[
        (predictions["gap_days"] >= MIN_GAP_DAYS) & predictions["is_champion"]
    ]
    return {
        "schema": SCHEMA_VERSION,
        "generated_at": now.astimezone(TIMEZONE).date().isoformat(),
        "rules": {
            "min_gap_days": MIN_GAP_DAYS,
            "gate_min_reviews": gate.MIN_REVIEWS,
            "gate_confidence": gate.CONFIDENCE,
            "gate_calibration_tolerance": gate.CALIBRATION_TOLERANCE,
            "fit_min_reviews": DEFAULT_MIN_REVIEWS,
            "min_week_reviews": MIN_WEEK_REVIEWS,
        },
        "totals": {
            "usable_reviews": int(usable["review_id"].nunique()),
            "first_week": weekly_first(usable),
        },
        "models": [
            {
                "version": m.version,
                "kind": m.kind,
                "role": m.role.value,
                "registered": m.created_at.astimezone(TIMEZONE).date().isoformat(),
                "trained_on_reviews": m.trained_on_reviews,
            }
            for m in models
        ],
        "weeks": weekly(predictions),
        "calibration": calibration(predictions, active),
        "gate": [
            {
                "challenger": d.challenger,
                "kind": d.kind,
                "promote": d.promote,
                "n": d.n,
                "champion_log_loss": _rounded(d.champion_log_loss),
                "challenger_log_loss": _rounded(d.challenger_log_loss),
                "champion_rmse_bins": _rounded(d.champion_rmse_bins),
                "challenger_rmse_bins": _rounded(d.challenger_rmse_bins),
                "p_better": _rounded(d.p_better, 3),
                "reasons": d.reasons,
            }
            for d in decisions
        ],
        "promotions": [
            {
                "date": p.promoted_at.astimezone(TIMEZONE).date().isoformat(),
                "version": p.version,
                "previous_version": p.previous_version,
                "reason": "gate" if p.reason == "gate" else "manual",
            }
            for p in promotions
        ],
    }


def publish(
    content: bytes, repository: str, branch: str, path: str, token: str
) -> bool:
    """Commit `content` to `path` on `branch` through the GitHub contents API (the
    trainer image has no git). Returns False when the file is already current."""
    url = f"https://api.github.com/repos/{repository}/contents/{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "forglemmigej-report",
    }

    def call(method: str, url: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(url, data, headers, method=method)
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    sha = None
    try:
        current = call("GET", f"{url}?ref={branch}")
        if base64.b64decode(current["content"]) == content:
            return False
        sha = current["sha"]
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
    body = {
        "message": f"Model report {datetime.now(TIMEZONE):%Y-%m-%d}",
        "content": base64.b64encode(content).decode(),
        "branch": branch,
    } | ({"sha": sha} if sha else {})
    call("PUT", url, body)
    return True


def render(report: dict) -> bytes:
    return (json.dumps(report, indent=1, sort_keys=False) + "\n").encode()


def publish_from_environment(report: dict) -> str:
    """Publish if a token is configured (REPORT_GITHUB_TOKEN); never raises, so a
    GitHub outage cannot fail the nightly training."""
    token = os.environ.get("REPORT_GITHUB_TOKEN")
    if not token:
        return "skipped: no REPORT_GITHUB_TOKEN"
    repository = os.environ.get("REPORT_REPOSITORY", "b-pasztor/forglemmigej")
    branch = os.environ.get("REPORT_BRANCH", "report-data")
    try:
        changed = publish(render(report), repository, branch, "report.json", token)
    except (urllib.error.URLError, OSError, ValueError, KeyError) as error:
        return f"failed: {error}"
    return "published" if changed else "unchanged"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", help="write the JSON here")
    parser.add_argument("--publish", action="store_true", help="commit it to GitHub")
    args = parser.parse_args(argv)

    from forglemmigej_api.db import SessionLocal

    with SessionLocal().begin() as session:
        report = build_report(session, datetime.now(UTC))
    if args.out:
        with open(args.out, "wb") as file:
            file.write(render(report))
    if args.publish:
        print(publish_from_environment(report))
    if not (args.out or args.publish):
        print(render(report).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
