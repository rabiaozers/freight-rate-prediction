"""Reproduce the model comparison: baselines vs final model on time-based folds and the cold-start test.

Usage:
    python run_validation.py --data-dir data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from freight.data import load_inputs
from freight.features import build_mi_daily, clean_features
from freight.model import FreightRateModel
from freight.validation import (FOLDS, evaluate, evaluate_cold_start, global_rpm_baseline,
                                lane_market_baseline)

MODELS = {
    "global rate-per-mile baseline": global_rpm_baseline,
    "lane median + market + trend baseline": lane_market_baseline,
    "LightGBM, detrended (final)": lambda train, test: FreightRateModel().fit(train).predict(test),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    raw = load_inputs(Path(args.data_dir))
    mi_daily = build_mi_daily(raw["train"], raw["validation"])
    train = clean_features(raw["train"], mi_daily)

    rows = []
    for name, fn in MODELS.items():
        folds = evaluate(train, fn)
        cold = evaluate_cold_start(train, fn)
        for view in ("raw", "clean"):
            f = folds[folds["view"] == view].set_index("fold")
            rows.append({
                "model": name, "view": view,
                **{f"MAPE {start[5:7]}-{end[5:7]}": f.loc[start, "MAPE"] for start, end in FOLDS},
                "MAPE mean": f["MAPE"].mean(),
                "MAPE cold-start": cold.loc[cold["view"] == view, "MAPE"].mean(),
                "median bias (last fold)": f.loc[FOLDS[-1][0], "median_bias"],
                "MAE mean": f["MAE"].mean(), "RMSE mean": f["RMSE"].mean(),
            })
        print(f"done: {name}")

    summary = pd.DataFrame(rows)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "validation_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
