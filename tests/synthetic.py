"""Small synthetic dataset with the assessment's schema, used only by the tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _haversine(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, [a1, o1, a2, o2])
    h = np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin((o2 - o1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(h))


def make_frames(n_train: int = 6000, n_valid: int = 12000, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    names = ["Lexington", "Fort Wayne"] + [f"City{i}" for i in range(28)]
    city = pd.DataFrame({"lat": rng.uniform(28, 44, len(names)), "lon": rng.uniform(-121, -70, len(names))}, index=names)
    city.loc["Lexington"] = [38.0, -84.5]
    city.loc["Fort Wayne"] = [41.0, -85.1]

    def make(dates, prefix, with_target):
        n = len(dates)
        p = rng.integers(0, len(names), n)
        d = (p + rng.integers(1, len(names), n)) % len(names)
        p[:30], d[:30] = 0, 1
        P, D = city.iloc[p], city.iloc[d]
        dist = np.clip(_haversine(P.lat.values, P.lon.values, D.lat.values, D.lon.values) * 1.18, 70, None)
        eq = rng.choice(["Dry Van", "Reefer", "Flatbed"], n, p=[0.6, 0.25, 0.15])
        weight = rng.normal(31500, 8000, n).clip(5000, 47500)
        t = (dates - pd.Timestamp("2025-01-01")).days.values
        mi = 1.08 + 0.2 * np.sin(2 * np.pi * (t - 60) / 365) + rng.normal(0, 0.02, n)
        rpm = (1.6 + 120 / dist) * mi ** 0.15 * np.exp(0.0065 * t / 30) * rng.lognormal(0, 0.03, n)
        df = pd.DataFrame({
            "load_id": [f"{prefix}-{i:06d}" for i in range(1, n + 1)],
            "pickup": P.index, "delivery": D.index,
            "pickup_lat": P.lat.values, "pickup_lon": P.lon.values,
            "delivery_lat": D.lat.values, "delivery_lon": D.lon.values,
            "distance": dist.round(1), "equipment": eq, "weight": weight.round(),
            "date": dates, "market_index": mi.round(5), "quote_signal": (rpm * rng.normal(1, 0.1, n)).round(5),
        })
        df.loc[rng.choice(n, n // 100, replace=False), "weight"] *= -1
        df.loc[rng.choice(n, n // 100, replace=False), "weight"] = np.nan
        df.loc[rng.choice(n, n // 100, replace=False), "market_index"] = np.nan
        if with_target:
            rate = rpm * dist
            bad = rng.choice(n, n // 70, replace=False)
            rate[bad] *= np.where(rng.random(len(bad)) < 0.5, 3.5, 1 / 3.5)
            df["posted_rate"] = rate.round(2)
        return df

    train_dates = pd.DatetimeIndex(np.sort(rng.choice(pd.date_range("2025-01-01", "2025-10-31"), n_train)))
    valid_dates = pd.DatetimeIndex(np.sort(rng.choice(pd.date_range("2025-11-01", "2025-12-31"), n_valid)))
    december = pd.DataFrame({
        "pickup": "Lexington", "delivery": "Fort Wayne", "distance": 360, "equipment": "Dry Van",
        "weight": 32000, "date": pd.date_range("2025-12-01", "2025-12-31"), "predicted_rate": np.nan,
    })
    return {"train": make(train_dates, "TR", True), "validation": make(valid_dates, "TE", False), "december": december}


def write_files(folder: Path, **kw) -> None:
    """Write the frames with the hyphenated names some downloads use, to exercise name discovery."""
    frames = make_frames(**kw)
    folder.mkdir(parents=True, exist_ok=True)
    fmt = lambda df: df.assign(date=pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"))
    fmt(frames["train"]).to_csv(folder / "train-test.csv", index=False)
    fmt(frames["validation"]).to_csv(folder / "validation.csv", index=False)
    fmt(frames["december"]).to_csv(folder / "december-chart-inputs.csv", index=False)
    ids = frames["validation"]["load_id"].sample(frac=1, random_state=1)
    pd.DataFrame({"load_id": ids, "predicted_rate": np.nan}).to_csv(
        folder / "validation-predictions-template.csv", index=False)
