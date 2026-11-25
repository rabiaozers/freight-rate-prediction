"""Time-based validation that mirrors the real task (learn through October, predict two months ahead).

- Expanding window, three folds, two-month horizon. No random K-fold: it would mix
  future market information and same-day loads into training.
- Outliers are detected and removed inside each fold's training part only.
- Test folds are scored twice: "raw" (as the grader will see labels) and "clean"
  (corrupted labels excluded) - model selection uses the clean view.
- Cold-start test: 8 random cities removed from training; scored on test lanes with
  exactly one unseen endpoint, as in validation.csv (12% of rows).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .features import DIST_BINS, outlier_mask

FOLDS = [("2025-05-01", "2025-07-01"), ("2025-07-01", "2025-09-01"), ("2025-09-01", "2025-11-01")]

FitPredict = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


def temporal_folds(df: pd.DataFrame):
    for start, end in FOLDS:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        yield start, df.index[df["date"] < start_ts], df.index[(df["date"] >= start_ts) & (df["date"] < end_ts)]


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = y - p
    return {"MAPE": np.abs(err / y).mean(), "MAE": np.abs(err).mean(), "RMSE": np.sqrt((err ** 2).mean()),
            "median_bias": np.median(p / y - 1)}


def _score(test: pd.DataFrame, pred: np.ndarray, clean: np.ndarray, **tags) -> list[dict]:
    y = test["posted_rate"].to_numpy()
    return [{**tags, "view": view, "n": int(m.sum()), **metrics(y[m], pred[m])}
            for view, m in (("raw", np.ones(len(y), bool)), ("clean", clean))]


def evaluate(df: pd.DataFrame, fit_predict: FitPredict) -> pd.DataFrame:
    rows = []
    for start, tr_idx, te_idx in temporal_folds(df):
        train = df.loc[tr_idx]
        train = train[~outlier_mask(train)]                       # training part only
        test = df.loc[te_idx]
        pred = np.asarray(fit_predict(train, test), float)
        rows += _score(test, pred, (~outlier_mask(test)).to_numpy(), fold=start)
    return pd.DataFrame(rows)


def evaluate_cold_start(df: pd.DataFrame, fit_predict: FitPredict, n_cities: int = 8,
                        seeds=(0, 1, 2)) -> pd.DataFrame:
    start, end = pd.Timestamp(FOLDS[-1][0]), pd.Timestamp(FOLDS[-1][1])
    train_all = df[df["date"] < start]
    test_all = df[(df["date"] >= start) & (df["date"] < end)]
    clean_all = ~outlier_mask(test_all)
    cities = sorted(set(df["pickup"]) | set(df["delivery"]))
    rows = []
    for seed in seeds:
        cold = set(np.random.default_rng(seed).choice(cities, n_cities, replace=False))
        train = train_all[~(train_all["pickup"].isin(cold) | train_all["delivery"].isin(cold))]
        train = train[~outlier_mask(train)]
        sel = test_all["pickup"].isin(cold) ^ test_all["delivery"].isin(cold)
        test = test_all[sel]
        pred = np.asarray(fit_predict(train, test), float)
        rows += _score(test, pred, clean_all[sel].to_numpy(), seed=seed)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- baselines
def global_rpm_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return test["distance"].to_numpy() * (train["posted_rate"] / train["distance"]).median()


def lane_market_baseline(train: pd.DataFrame, test: pd.DataFrame, min_n: int = 5) -> np.ndarray:
    """Lane x equipment median rate per mile (fallback: equipment x distance band),
    corrected by OLS on log(mi_daily) and the monthly trend."""
    def keyed(d):
        k = d[["pickup", "delivery", "equipment", "distance"]].copy()
        k["band"] = pd.cut(k["distance"], DIST_BINS).astype(str)
        return k

    tr = keyed(train).assign(lrpm=np.log(train["posted_rate"] / train["distance"]).to_numpy())
    lane = tr.groupby(["pickup", "delivery", "equipment"])["lrpm"].agg(["median", "size"])
    lane = lane.loc[lane["size"] >= min_n, "median"].rename("lane")
    band = tr.groupby(["equipment", "band"])["lrpm"].median().rename("fallback")

    def base(d):
        k = keyed(d).merge(lane, left_on=["pickup", "delivery", "equipment"], right_index=True, how="left")
        k = k.merge(band, left_on=["equipment", "band"], right_index=True, how="left")
        return k["lane"].fillna(k["fallback"]).to_numpy()

    def design(d):
        t = ((d["date"] - pd.Timestamp("2025-01-01")).dt.days / 30).to_numpy()
        return np.column_stack([np.ones(len(d)), np.log(d["mi_daily"]), t])

    resid = tr["lrpm"].to_numpy() - base(train)
    coef, *_ = np.linalg.lstsq(design(train), resid, rcond=None)
    return test["distance"].to_numpy() * np.exp(base(test) + design(test) @ coef)
