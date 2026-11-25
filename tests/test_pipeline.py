"""Tests for the leakage rules and the deliverable formats (synthetic data, runs in under a minute)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from freight.features import add_coordinates, build_mi_daily, city_coordinates, clean_features, outlier_mask  # noqa: E402
from freight.model import FreightRateModel  # noqa: E402
from freight.validation import temporal_folds  # noqa: E402
from synthetic import make_frames, write_files  # noqa: E402


@pytest.fixture(scope="module")
def frames():
    f = make_frames()
    mi = build_mi_daily(f["train"], f["validation"])
    return {**f, "mi": mi, "train_c": clean_features(f["train"], mi), "valid_c": clean_features(f["validation"], mi)}


def test_clean_features(frames):
    c = frames["train_c"]
    assert "quote_signal" not in c and "market_index" not in c
    assert (c["weight"].dropna() > 0).all()
    assert c["mi_daily"].notna().all() and frames["valid_c"]["mi_daily"].notna().all()


def test_outlier_mask_flags_corrupted_rates(frames):
    c = frames["train_c"]
    rpm = c["posted_rate"] / c["distance"]
    corrupted = (rpm / rpm.groupby(c["equipment"]).transform("median")).pipe(lambda r: (r > 2.5) | (r < 0.4))
    flagged = outlier_mask(c)
    assert flagged[corrupted].mean() > 0.9           # catches the injected x3.5 / /3.5 rows
    assert flagged[~corrupted].mean() < 0.01         # and leaves clean rows alone


def test_temporal_folds_never_train_on_the_future(frames):
    c = frames["train_c"]
    for _, tr_idx, te_idx in temporal_folds(c):
        assert c.loc[tr_idx, "date"].max() < c.loc[te_idx, "date"].min()


def test_december_coordinates_come_from_training_map(frames):
    coords = city_coordinates(frames["train"])
    dec = add_coordinates(frames["december"], coords)
    assert dec[["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]].notna().all().all()
    with pytest.raises(ValueError):
        add_coordinates(frames["december"].assign(pickup="Atlantis"), coords)


def test_model_predictions_are_positive_and_trend_is_recovered(frames):
    c = frames["train_c"]
    model = FreightRateModel().fit(c[~outlier_mask(c)])
    pred = model.predict(frames["valid_c"])
    assert len(pred) == len(frames["valid_c"]) and np.isfinite(pred).all() and (pred > 0).all()
    assert abs(model.trend - 0.0065) < 0.003           # synthetic data uses 0.65% per month


def test_pipeline_writes_valid_deliverables(tmp_path):
    data = tmp_path / "data"
    write_files(data)
    out = tmp_path / "outputs"
    subprocess.run([sys.executable, str(ROOT / "run_pipeline.py"), "--data-dir", str(data), "--output-dir", str(out),
                    "--no-score"], check=True, cwd=ROOT)
    sub = pd.read_csv(out / "validation_predictions.csv")
    assert list(sub.columns) == ["load_id", "predicted_rate"]
    assert len(sub) == 12_000 and sub["load_id"].is_unique
    assert set(sub["load_id"]) == {f"TE-{i:06d}" for i in range(1, 12_001)}
    assert (sub["predicted_rate"] > 0).all()
    template = pd.read_csv(data / "validation-predictions-template.csv")
    assert sub["load_id"].tolist() == template["load_id"].tolist()   # template order preserved
    dec = pd.read_csv(out / "december_chart_inputs.csv")
    assert list(dec.columns) == ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    assert len(dec) == 31 and (dec["predicted_rate"] > 0).all()
