from datetime import UTC, datetime, timedelta

import pytest

pd = pytest.importorskip("pandas")  # the ml dependency group
np = pytest.importorskip("numpy")

from forglemmigej_api.training.evaluate import (
    calibration_table,
    chronological_split,
    rmse_bins,
    score,
)


def test_perfectly_calibrated_predictions_have_no_bin_error() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 200_000)
    y = rng.uniform(size=p.size) < p
    assert rmse_bins(y, p) < 0.01
    assert rmse_bins(y, np.clip(p + 0.2, 0, 1)) > 0.1  # overconfident


def test_score_rewards_better_predictions() -> None:
    y = np.array([1, 1, 1, 0, 0])
    good = score(y, [0.9, 0.8, 0.9, 0.2, 0.1])
    bad = score(y, [0.5, 0.5, 0.5, 0.5, 0.5])
    assert good.log_loss < bad.log_loss
    assert good.auc == 1.0 and bad.auc == 0.5
    assert good.n == 5


def test_calibration_table_bins() -> None:
    table = calibration_table([1, 0, 1, 1], [0.92, 0.91, 0.12, 0.13], bins=10)
    assert table["n"].tolist() == [2, 2]
    assert table["actual"].tolist() == [1.0, 0.5]


def test_chronological_split_never_puts_older_reviews_in_test() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    data = pd.DataFrame(
        {
            "user": ["a"] * 10 + ["b"] * 10,
            "reviewed_at": [t0 + timedelta(days=i) for i in range(10)]
            + [t0 + timedelta(days=100 + i) for i in range(10)],
        }
    )
    train, test = chronological_split(data, 0.2, group="user")
    for user in "ab":
        assert (
            train[train.user == user].reviewed_at.max()
            < test[test.user == user].reviewed_at.min()
        )
    assert len(test) == 4
