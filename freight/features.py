"""Feature cleaning shared by every dataset (train, validation, December scenario).

Rule: transformations are applied identically everywhere; rows are only ever
removed from *training* data (see ``outlier_mask``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DIST_BINS = [0, 250, 500, 1000, 2000, 4000]
T_START = pd.Timestamp("2025-01-01")
EQUIPMENT = pd.CategoricalDtype(["Dry Van", "Flatbed", "Reefer"])
COORD_COLS = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]


def build_mi_daily(*frames: pd.DataFrame) -> pd.Series:
    """Daily market index = mean ``market_index`` over all loads of that date.

    Uses the input column only (never the target). ~98% of market_index variance is
    between days, so the daily mean keeps the signal, fills missing values, and gives
    the December scenario (which has no market_index column) a value for each date.
    Assumption: the day's market index is known at quote time.
    """
    df = pd.concat([f[["date", "market_index"]] for f in frames], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df.groupby("date")["market_index"].mean()


def city_coordinates(train: pd.DataFrame) -> pd.DataFrame:
    """City -> (lat, lon) as given in the training data.

    Coordinates are anonymised (shifted) but consistent per city, so they are used as
    location features only and never replaced with real-world coordinates.
    """
    cols = ["city", "lat", "lon"]
    cities = pd.concat([
        train[["pickup", "pickup_lat", "pickup_lon"]].set_axis(cols, axis=1),
        train[["delivery", "delivery_lat", "delivery_lon"]].set_axis(cols, axis=1),
    ])
    return cities.drop_duplicates("city").set_index("city")[["lat", "lon"]]


def add_coordinates(df: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """Fill lat/lon for inputs that lack them (the December scenario) from the training map."""
    out = df.copy()
    for side in ("pickup", "delivery"):
        out[f"{side}_lat"] = out[side].map(coords["lat"])
        out[f"{side}_lon"] = out[side].map(coords["lon"])
    missing = out[COORD_COLS].isna().any(axis=1)
    if missing.any():
        unknown = sorted(set(out.loc[missing, "pickup"]) | set(out.loc[missing, "delivery"]))
        raise ValueError(f"No training coordinates for: {unknown}")
    return out


def clean_features(df: pd.DataFrame, mi_daily: pd.Series) -> pd.DataFrame:
    """Same feature cleaning for every dataset.

    - weight: negative values mirror the positive distribution -> sign error -> abs();
      0 is treated as missing; missing values stay NaN (LightGBM handles them natively).
    - market_index -> mi_daily (daily mean, see ``build_mi_daily``).
    - quote_signal is dropped: in five training months ~15% of its rows equal the
      target rate per mile exactly (a month-dependent target leak), the validation
      months look like the non-leaking regime, and the December file lacks it.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["weight"] = out["weight"].abs().replace(0, np.nan)
    out["mi_daily"] = out["date"].map(mi_daily)
    if out["mi_daily"].isna().any():
        raise ValueError("mi_daily missing for some dates")
    return out.drop(columns=["quote_signal", "market_index"], errors="ignore")


def outlier_mask(df: pd.DataFrame, z: float = 4.0) -> pd.Series:
    """Flag impossible rates: robust z-score of log(rate per mile) within equipment x distance band.

    The robust-z distribution has an empty gap between 4 and 6, so any threshold in that
    range selects the same rows. Use on TRAINING data only; validation rows are never dropped.
    """
    lr = np.log(df["posted_rate"] / df["distance"])
    groups = [df["equipment"], pd.cut(df["distance"], DIST_BINS)]
    med = lr.groupby(groups, observed=True).transform("median")
    mad = (lr - med).abs().groupby(groups, observed=True).transform("median")
    return ((lr - med) / (1.4826 * mad)).abs() > z


def months_since_start(df: pd.DataFrame) -> np.ndarray:
    return ((df["date"] - T_START).dt.days / 30).to_numpy()


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Model matrix. No city IDs: 12% of validation rows involve a city never seen in
    training, so location is learned from coordinates, which exist for those cities too."""
    return pd.DataFrame({
        "pickup_lat": df["pickup_lat"], "pickup_lon": df["pickup_lon"],
        "delivery_lat": df["delivery_lat"], "delivery_lon": df["delivery_lon"],
        "log_dist": np.log(df["distance"]),
        "equipment": df["equipment"].astype(EQUIPMENT),
        "weight": df["weight"],
        "log_mi": np.log(df["mi_daily"]),
        "dow": df["date"].dt.dayofweek,
    }, index=df.index)
