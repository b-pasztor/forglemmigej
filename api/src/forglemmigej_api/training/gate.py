"""The champion/challenger gate: may a challenger take over scheduling?

The evidence is the shadow predictions: on the reviews that both the champion
and the challenger predicted (and only those at least a day after the card's
previous review), the challenger must be

1. able to schedule (FSRS; LightGBM only predicts, so it is compared but never
   promoted),
2. judged on enough reviews (MIN_REVIEWS),
3. better on log loss, and convincingly so: in a bootstrap over cards the
   challenger wins in at least CONFIDENCE of the resamples (one lucky week is
   not enough),
4. not worse calibrated (binned RMSE at most CALIBRATION_TOLERANCE above the
   champion's), because the scheduler acts on the probability around 90%.

Shadow predictions only exist from a challenger's registration on, and it was
trained on reviews before that, so this is always a test on unseen, later data.

On promotion the old champion retires, the cached FSRS state of every card is
rebuilt by replaying the review log with the new parameters (so the cache stays
"what the champion would have computed"), and the change is recorded in
`model_promotions`. A manual rollback uses the same path:

    uv run --group ml python -m forglemmigej_api.training.gate --report
    uv run --group ml python -m forglemmigej_api.training.gate --promote fsrs6-default --reason "rollback"
"""

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from forglemmigej_api.models import (
    Card,
    CardStatus,
    Grade,
    MemoryModel,
    ModelPromotion,
    ModelRole,
    Review,
)
from forglemmigej_api.scheduling import FsrsModel, apply_review
from forglemmigej_api.training.evaluate import EPSILON, score

MIN_REVIEWS = 300
CONFIDENCE = 0.95
CALIBRATION_TOLERANCE = 0.01
BOOTSTRAP_SAMPLES = 2000
MIN_GAP_DAYS = 1.0
SCHEDULING_KINDS = {"fsrs"}


@dataclass
class Decision:
    challenger: str
    kind: str
    promote: bool
    n: int = 0
    champion_log_loss: float = float("nan")
    challenger_log_loss: float = float("nan")
    champion_rmse_bins: float = float("nan")
    challenger_rmse_bins: float = float("nan")
    p_better: float = float("nan")  # share of bootstrap resamples it wins
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _log_losses(recalled: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    p = np.clip(predicted, EPSILON, 1 - EPSILON)
    return -(recalled * np.log(p) + (1 - recalled) * np.log(1 - p))


def bootstrap_p_better(
    cards: np.ndarray,
    champion_loss: np.ndarray,
    challenger_loss: np.ndarray,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> float:
    """Share of bootstrap resamples in which the challenger has the lower mean log
    loss. Resamples whole cards, not single reviews: reviews of the same card are
    not independent, and treating them as such would overstate the confidence."""
    frame = pd.DataFrame({"card": cards, "diff": champion_loss - challenger_loss})
    per_card = frame.groupby("card")["diff"].agg(["sum", "size"])
    sums, sizes = per_card["sum"].to_numpy(), per_card["size"].to_numpy()
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(sums), size=(samples, len(sums)))
    mean_diff = sums[picks].sum(axis=1) / sizes[picks].sum(axis=1)
    return float((mean_diff > 0).mean())


def decide(
    predictions: pd.DataFrame, champion: str, challenger: str, kind: str
) -> Decision:
    """Apply the gate to one challenger. `predictions` has one row per (review,
    model): review_id, card_id, model_version, predicted_recall, recalled,
    gap_days (see nightly.shadow_predictions)."""
    eligible = predictions[
        (predictions["gap_days"] >= MIN_GAP_DAYS)
        & predictions["model_version"].isin([champion, challenger])
    ]
    pair = (
        eligible.pivot_table(
            index=["review_id", "card_id", "recalled"],
            columns="model_version",
            values="predicted_recall",
        )
        .reindex(columns=[champion, challenger])
        .dropna()
        .reset_index()
    )
    decision = Decision(challenger, kind, promote=False, n=len(pair))
    if kind not in SCHEDULING_KINDS:
        decision.reasons.append(f"{kind} models cannot schedule")
    if len(pair) < MIN_REVIEWS:
        decision.reasons.append(f"{len(pair)} shared reviews < {MIN_REVIEWS}")
    if len(pair) < 2:
        return decision

    recalled = pair["recalled"].to_numpy(float)
    ours, theirs = pair[champion].to_numpy(), pair[challenger].to_numpy()
    champion_scores, challenger_scores = score(recalled, ours), score(recalled, theirs)
    decision.champion_log_loss = champion_scores.log_loss
    decision.challenger_log_loss = challenger_scores.log_loss
    decision.champion_rmse_bins = champion_scores.rmse_bins
    decision.challenger_rmse_bins = challenger_scores.rmse_bins
    decision.p_better = bootstrap_p_better(
        pair["card_id"].to_numpy(),
        _log_losses(recalled, ours),
        _log_losses(recalled, theirs),
    )
    if decision.p_better < CONFIDENCE:
        decision.reasons.append(
            f"better log loss in {decision.p_better:.0%} of resamples < {CONFIDENCE:.0%}"
        )
    if (
        decision.challenger_rmse_bins
        > decision.champion_rmse_bins + CALIBRATION_TOLERANCE
    ):
        decision.reasons.append(
            f"worse calibrated (RMSE {decision.challenger_rmse_bins:.3f} vs "
            f"{decision.champion_rmse_bins:.3f})"
        )
    decision.promote = not decision.reasons
    return decision


def rebuild_card_states(session: Session, model: FsrsModel) -> int:
    """Replay every reviewed card's log with `model` and store the result as the
    card's cached state. Suspended cards stay suspended."""
    reviews = session.execute(
        select(Review.card_id, Review.reviewed_at, Review.grade).order_by(
            Review.card_id, Review.reviewed_at, Review.id
        )
    ).all()
    by_card: dict[int, list[tuple[datetime, int]]] = {}
    for card_id, reviewed_at, grade in reviews:
        # FSRS insists on exactly UTC.
        by_card.setdefault(card_id, []).append((reviewed_at.astimezone(UTC), grade))
    for card_id, history in by_card.items():
        card = session.get(Card, card_id, with_for_update=True)
        state = None
        for reviewed_at, grade in history:
            state, due_at, status = apply_review(
                model, state, Grade(grade), reviewed_at
            )
        card.scheduler_state, card.due_at = state, due_at
        if card.status != CardStatus.SUSPENDED:
            card.status = status
    session.flush()
    return len(by_card)


def promote(
    session: Session,
    version: str,
    reason: str,
    evaluation: dict | None = None,
) -> ModelPromotion:
    """Make `version` the champion: the old champion retires and card states are
    rebuilt with the new parameters, all in the caller's transaction."""
    new = session.scalars(
        select(MemoryModel).where(MemoryModel.version == version).with_for_update()
    ).one()
    if new.kind not in SCHEDULING_KINDS:
        raise ValueError(f"{version} is a {new.kind} model and cannot schedule")
    old = session.scalars(
        select(MemoryModel)
        .where(MemoryModel.role == ModelRole.CHAMPION)
        .with_for_update()
    ).one()
    if old.id == new.id:
        raise ValueError(f"{version} is already the champion")
    old.role = ModelRole.RETIRED
    session.flush()  # one champion at a time (unique index)
    new.role = ModelRole.CHAMPION
    session.flush()
    rebuilt = rebuild_card_states(
        session, FsrsModel(new.version, tuple(new.parameters))
    )
    promotion = ModelPromotion(
        version=new.version,
        previous_version=old.version,
        reason=reason,
        evaluation=evaluation or {},
        cards_rebuilt=rebuilt,
    )
    session.add(promotion)
    session.flush()
    return promotion


def run_gate(
    session: Session, predictions: pd.DataFrame, *, apply: bool = True
) -> tuple[list[Decision], ModelPromotion | None]:
    """Judge every challenger; promote the best one that passes (lowest log loss)."""
    models = session.execute(
        select(MemoryModel.version, MemoryModel.kind, MemoryModel.role).where(
            MemoryModel.role.in_([ModelRole.CHAMPION, ModelRole.CHALLENGER])
        )
    ).all()
    champion = next(v for v, _, role in models if role == ModelRole.CHAMPION)
    decisions = [
        decide(predictions, champion, version, kind)
        for version, kind, role in models
        if role == ModelRole.CHALLENGER
    ]
    winners = sorted(
        (d for d in decisions if d.promote), key=lambda d: d.challenger_log_loss
    )
    if not (apply and winners):
        return decisions, None
    best = winners[0]
    return decisions, promote(session, best.challenger, "gate", best.as_dict())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--report", action="store_true", help="judge, change nothing")
    action.add_argument("--promote", metavar="VERSION", help="manual promotion")
    parser.add_argument("--reason", default="", help="why (manual promotion)")
    args = parser.parse_args(argv)

    from forglemmigej_api.db import SessionLocal
    from forglemmigej_api.training.nightly import shadow_predictions

    with SessionLocal().begin() as session:
        if args.promote:
            promotion = promote(session, args.promote, f"manual: {args.reason}".strip())
            print(
                f"{promotion.previous_version} -> {promotion.version}, "
                f"{promotion.cards_rebuilt} cards rebuilt"
            )
            return 0
        since = datetime(1970, 1, 1, tzinfo=UTC)
        decisions, _ = run_gate(
            session, shadow_predictions(session, since), apply=False
        )
        for decision in decisions:
            print(json.dumps(decision.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
