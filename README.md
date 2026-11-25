# Freight Rate Prediction

Predicts the posted rate of US truckload shipments for November–December 2025 from January–October 2025 history.
The final model is a LightGBM regressor on detrended log rate per mile, validated with an expanding-window time split
and a cold-start (unseen city) test.

| Model | MAPE, last fold (Sep–Oct) | MAPE, unseen cities | MAPE, mean of 3 folds |
|---|---|---|---|
| Global rate-per-mile baseline | 9.27% | 8.92% | 9.37% |
| Lane median + market + trend baseline | 3.65% | 3.97% | 4.17% |
| **LightGBM, detrended (final)** | **2.03%** | **2.12%** | **2.84%** |

MAPE on clean test labels (the 1.4% of rows with corrupted rates excluded). On raw labels the final model scores 4.5% / 4.4% / 5.2%:
the corrupted rows add ~2.4 points that no model can remove. Reproduce with `python run_validation.py`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# put the provided CSV files in data/ and score.py in the repository root (see data/README.md)
python run_pipeline.py          # trains, writes outputs/, runs score.py -> outputs/scorer_results/candidate_december.png
python run_validation.py        # model comparison table -> outputs/validation_summary.csv (~3 min)
pytest -q                       # leakage rules and deliverable formats on synthetic data (~10 s)
```

Outputs:

- `outputs/validation_predictions.csv` – 12,000 rows, `load_id,predicted_rate`, in the template's row order
- `outputs/december_chart_inputs.csv` – the provided December file with `predicted_rate` filled
- `outputs/scorer_results/candidate_december.png` – chart produced by the provided `score.py`

## Repository layout

```
freight/
  data.py         input discovery (underscore or hyphen file names) and loading
  features.py     cleaning shared by train / validation / December; outlier rule; model features
  model.py        FreightRateModel: OLS trend removal + LightGBM on log rate per mile
  validation.py   time-based folds, cold-start test, metrics, baselines
run_pipeline.py   end-to-end: clean -> train on Jan–Oct -> deliverables -> score.py
run_validation.py reproduces the comparison table above
tests/            synthetic-data tests for the leakage rules and output formats
notebooks/        exploratory analysis, experiments, and quote_signal leak diagnosis (with outputs)
reports/          technical report (PDF and DOCX)
```

## Key decisions

**Validation.** The real task is "learn until October, predict the next two months", so validation uses an expanding
window with three folds and a two-month horizon (train Jan–Apr → test May–Jun, Jan–Jun → Jul–Aug, Jan–Aug → Sep–Oct).
Random K-fold would let same-day loads and future market information into training. Everything learned from the target
(outlier rule, trend slope) is fitted inside each training fold only. Because 12% of validation rows involve a city that
never appears in training, a cold-start test removes 8 random cities from training and scores lanes with exactly one
unseen endpoint.

**Data cleaning.** Transformations are identical for every dataset; rows are removed from training data only.

| Issue | Evidence | Decision |
|---|---|---|
| `quote_signal` | In Jan, Feb, Mar, Jun and Sep ~15% of rows equal the target rate per mile exactly; in other months it is 4–12% off. Validation months match the non-leaking regime. Absent from the December file. | Dropped (month-dependent target leak) |
| `market_index` missing, absent in December | ~98% of its variance is between days | Replaced by its daily mean `mi_daily` |
| Negative `weight` | Absolute values match the positive distribution | `abs()`; missing values left to LightGBM |
| 677 rows (1.4%) with impossible rates | Robust z-score of log rate per mile has an empty gap between 4 and 6; symmetric ×0.2–0.5 / ×2–5 errors spread evenly over months | Removed from training only |
| Coordinates do not match real cities | Shifted, but each city has one consistent coordinate | Used as location features; no derived distances |

**Model.** Pricing is multiplicative (distance × rate × market), so the target is log(rate per mile). City IDs are not
used, so the model generalises to unseen cities through coordinates (cold-start MAPE 2.12% vs 2.03% for seen cities).
Trees cannot extrapolate a trend past October, so a monthly trend (0.65%/month, estimated by OLS controlling for distance
and market index) is removed from the target and added back at prediction time. Lane target encoding, per-equipment models,
an ensemble and six hyperparameter settings were tested; none moved results beyond ±0.05 points.

## Assumptions and limitations

- The day's market index is known at quote time; December values come from the `market_index` column of `validation.csv`
  (an input, never the target).
- Predictions are `exp(predicted log rate)`, a conditional median suited to relative-error metrics.
- Monthly price levels move ±1.5% beyond what the market index and trend explain, and Flatbed months are more volatile;
  ten months of data cannot capture seasonality for November–December.
- Fold-level bias (+4.8% for Jul–Aug) comes from short training windows in which market index and time rise together
  (correlation 0.86 for Jan–Jun), which inflates the estimated trend. On the full Jan–Oct window the correlation is −0.11
  and the estimates (elasticity 0.15, trend 0.65%/month) match the independent day-level analysis.
