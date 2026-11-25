"""Final model: LightGBM on detrended log(rate per mile).

Trees cannot extrapolate a trend beyond the last training month (October), so a linear
monthly trend is estimated from the training data, removed from the target before
fitting, and added back at prediction time.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import make_features, months_since_start

GBM_PARAMS = dict(
    n_estimators=800, learning_rate=0.03, num_leaves=63, min_child_samples=40,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    random_state=42, deterministic=True, force_row_wise=True, verbose=-1,
)


def log_rate_per_mile(df: pd.DataFrame) -> np.ndarray:
    return np.log(df["posted_rate"] / df["distance"]).to_numpy()


def trend_slope(train: pd.DataFrame) -> float:
    """Monthly trend from OLS: log(rpm) ~ 1 + log(distance) + log(mi_daily) + months.

    Controlling for the market index matters: market index and time are correlated,
    and without it the trend would absorb market moves.
    """
    design = np.column_stack([
        np.ones(len(train)), np.log(train["distance"]), np.log(train["mi_daily"]), months_since_start(train),
    ])
    coef, *_ = np.linalg.lstsq(design, log_rate_per_mile(train), rcond=None)
    return float(coef[-1])


class FreightRateModel:
    def __init__(self, params: dict | None = None):
        self.params = {**GBM_PARAMS, **(params or {})}
        self.trend: float | None = None
        self.gbm: lgb.LGBMRegressor | None = None

    def fit(self, train: pd.DataFrame) -> "FreightRateModel":
        self.trend = trend_slope(train)
        target = log_rate_per_mile(train) - self.trend * months_since_start(train)
        self.gbm = lgb.LGBMRegressor(**self.params).fit(make_features(train), target)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predicted rate in dollars. exp(log prediction) is a conditional median, which suits
        relative-error metrics; with residual sigma ~0.025 a smearing correction is <0.05%."""
        log_rpm = self.gbm.predict(make_features(df)) + self.trend * months_since_start(df)
        return df["distance"].to_numpy() * np.exp(log_rpm)

    def feature_importance(self) -> pd.Series:
        gain = pd.Series(self.gbm.booster_.feature_importance("gain"), index=self.gbm.feature_name_)
        return (gain / gain.sum()).sort_values(ascending=False)
