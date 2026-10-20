from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session

pd = pytest.importorskip("pandas")  # the ml dependency group

from forglemmigej_api.scheduling import LightGbmModel, active_models
from forglemmigej_api.training.pretrain_lightgbm import (
    FEATURES,
    register_challenger,
)

from .test_shadow import tiny_lightgbm


def test_a_new_pretrained_model_retires_the_previous_one(db: Session) -> None:
    text = tiny_lightgbm(FEATURES).model_to_string()
    data = pd.DataFrame(
        {
            "user": ["a", "b"],
            "reviewed_at": pd.to_datetime([datetime(2024, 1, 1, tzinfo=UTC)] * 2),
        }
    )
    register_challenger(db, text, data, "lightgbm-public-v1")
    register_challenger(db, text, data, "lightgbm-public-v2")

    _, challengers = active_models(db)
    assert [m.version for m in challengers] == ["lightgbm-public-v2"]
    assert isinstance(challengers[0], LightGbmModel)
    assert challengers[0].features == FEATURES
