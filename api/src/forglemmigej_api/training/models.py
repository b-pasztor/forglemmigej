"""Challenger recall models. Each predicts P(recalled) from the shared features."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

LABEL = "recalled"


def fit_logistic(train: pd.DataFrame, features: Sequence[str]) -> Pipeline:
    """Logistic regression: one weight per feature, easy to read.
    Missing values are filled with the median, then features are put on the
    same scale so the weights are comparable."""
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
    return model.fit(train[list(features)], train[LABEL])


def fit_lightgbm(
    train: pd.DataFrame,
    features: Sequence[str],
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> LGBMClassifier:
    """Gradient-boosted trees. The most recent slice of the training data is held
    out to decide when to stop adding trees (early stopping), so the model does
    not just memorise the training set."""
    train = train.sort_values("reviewed_at")
    cut = int(len(train) * (1 - validation_fraction))
    fit, valid = train.iloc[:cut], train.iloc[cut:]
    model = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=seed,
        verbose=-1,
    )
    model.fit(
        fit[list(features)],
        fit[LABEL],
        eval_set=[(valid[list(features)], valid[LABEL])],
        eval_metric="binary_logloss",
        callbacks=[early_stopping(50, verbose=False)],
    )
    return model


def predict(model, data: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    return model.predict_proba(data[list(features)])[:, 1]
