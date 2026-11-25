"""End-to-end pipeline: clean -> train the final model on Jan-Oct -> write both deliverables -> run score.py.

Usage:
    python run_pipeline.py --data-dir data --output-dir outputs
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from freight.data import FILES, find_file, load_inputs, require_file
from freight.features import add_coordinates, build_mi_daily, city_coordinates, clean_features, outlier_mask
from freight.model import FreightRateModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data", help="folder with the provided CSV files")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--score-script", default=None,
                        help="path to the provided score.py (default: ./score.py or <data-dir>/score.py)")
    parser.add_argument("--no-score", action="store_true", help="skip running score.py")
    args = parser.parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load and clean - identical feature transformations for every dataset
    raw = load_inputs(data_dir)
    mi_daily = build_mi_daily(raw["train"], raw["validation"])          # input column only, never the target
    coords = city_coordinates(raw["train"])
    train = clean_features(raw["train"], mi_daily)
    valid = clean_features(raw["validation"], mi_daily)
    december = clean_features(add_coordinates(raw["december"], coords), mi_daily)

    # 2) Rows are removed from TRAINING data only
    is_outlier = outlier_mask(train)
    train_fit = train[~is_outlier]
    print(f"Training rows: {len(train_fit):,} (removed {int(is_outlier.sum())} corrupted-rate rows)")

    # 3) Final model on all of Jan-Oct
    model = FreightRateModel().fit(train_fit)
    print(f"Trend: {model.trend:.4f} per month (~{model.trend * 12:.1%} per year)")
    print("Feature importance (gain share):", model.feature_importance().round(3).to_dict())

    valid_pred = model.predict(valid)
    dec_pred = model.predict(december)
    for name, pred, n in (("validation", valid_pred, len(valid)), ("december", dec_pred, len(december))):
        if len(pred) != n or not np.isfinite(pred).all() or (pred <= 0).any():
            raise ValueError(f"invalid {name} predictions")

    # 4) validation_predictions.csv - template row order, exactly load_id,predicted_rate
    template_path = find_file(data_dir, *FILES["template"])
    ids = pd.read_csv(template_path)["load_id"] if template_path else raw["validation"]["load_id"]
    pred_by_id = pd.Series(valid_pred, index=raw["validation"]["load_id"].to_numpy())
    submission = pd.DataFrame({"load_id": ids, "predicted_rate": ids.map(pred_by_id).round(2)})
    if submission["predicted_rate"].isna().any() or not submission["load_id"].is_unique:
        raise ValueError("submission has missing or duplicate ids")
    submission.to_csv(out_dir / "validation_predictions.csv", index=False)

    # 5) December file - the original file, only predicted_rate filled
    dec_out = pd.read_csv(require_file(data_dir, "december"))
    if not (pd.to_datetime(dec_out["date"]).to_numpy() == december["date"].to_numpy()).all():
        raise ValueError("December row order changed")
    dec_out["predicted_rate"] = dec_pred.round(2)
    dec_out.to_csv(out_dir / "december_chart_inputs.csv", index=False)
    corr = np.corrcoef(dec_pred, december["mi_daily"])[0, 1]
    print(f"December: ${dec_pred.min():.1f}-${dec_pred.max():.1f} (mean ${dec_pred.mean():.1f}), "
          f"corr with daily market index {corr:.2f}")

    # 6) Validate both files with the provided scorer and draw the December chart
    if args.no_score:
        return
    score = Path(args.score_script) if args.score_script else (find_file(Path("."), "score.py")
                                                               or find_file(data_dir, "score.py"))
    if score is None:
        print("score.py not found - skipped (pass --score-script)")
        return
    subprocess.run([sys.executable, str(score),
                    "--predictions", str(out_dir / "validation_predictions.csv"),
                    "--december-predictions", str(out_dir / "december_chart_inputs.csv"),
                    "--output-dir", str(out_dir / "scorer_results")], check=True)


if __name__ == "__main__":
    main()
