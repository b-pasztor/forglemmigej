"""How good is a recall model? Metrics shared by notebooks and the promotion gate.

- log loss: the main metric. Punishes confident mistakes (saying 95% for a card
  that was forgotten). Lower is better.
- Brier score: mean squared error of the probabilities. Lower is better.
- AUC: can the model rank remembered cards above forgotten ones? 0.5 is guessing.
- RMSE (bins): calibration error, as in the FSRS benchmark. Predictions are
  grouped into bins and each bin's average prediction is compared with how often
  those cards were actually recalled. Lower is better.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

EPSILON = 1e-6


@dataclass(frozen=True)
class Scores:
    log_loss: float
    brier: float
    auc: float
    rmse_bins: float
    n: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def calibration_table(y_true, y_pred, bins: int = 20) -> pd.DataFrame:
    """Per probability bin: mean prediction, actual recall rate and count."""
    frame = pd.DataFrame(
        {"y": np.asarray(y_true, float), "p": np.asarray(y_pred, float)}
    )
    frame["bin"] = np.minimum((frame["p"] * bins).astype(int), bins - 1)
    table = frame.groupby("bin").agg(
        predicted=("p", "mean"), actual=("y", "mean"), n=("y", "size")
    )
    return table.reset_index(drop=True)


def rmse_bins(y_true, y_pred, bins: int = 20) -> float:
    table = calibration_table(y_true, y_pred, bins)
    weights = table["n"] / table["n"].sum()
    return float(np.sqrt((weights * (table["predicted"] - table["actual"]) ** 2).sum()))


def score(y_true, y_pred) -> Scores:
    y_true = np.asarray(y_true, float)
    y_pred = np.clip(np.asarray(y_pred, float), EPSILON, 1 - EPSILON)
    return Scores(
        log_loss=float(log_loss(y_true, y_pred, labels=[0, 1])),
        brier=float(brier_score_loss(y_true, y_pred)),
        auc=float(roc_auc_score(y_true, y_pred))
        if len(set(y_true)) == 2
        else float("nan"),
        rmse_bins=rmse_bins(y_true, y_pred),
        n=len(y_true),
    )


def chronological_split(
    data: pd.DataFrame, test_fraction: float = 0.2, group: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Older reviews for training, the most recent ones for testing; never random.
    With `group` (e.g. "user"), each group is split at its own point in time."""
    if group is None:
        cutoff = data["reviewed_at"].quantile(1 - test_fraction)
        is_test = data["reviewed_at"] > cutoff
    else:
        cutoffs = data.groupby(group)["reviewed_at"].transform(
            lambda t: t.quantile(1 - test_fraction)
        )
        is_test = data["reviewed_at"] > cutoffs
    return data[~is_test], data[is_test]
