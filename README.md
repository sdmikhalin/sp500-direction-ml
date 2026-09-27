# Predicting the Direction of S&P 500 Daily Moves

Can next-day direction of the S&P 500 be predicted from its own price history using
standard technical indicators? This project runs that experiment carefully — with
time-series validation, a naive benchmark, and no shuffling — and finds that it
cannot. The negative result is the finding.

## TL;DR

| Model | Accuracy (test) | ROC-AUC (test) | ROC-AUC (CV) |
|---|---|---|---|
| Logistic Regression | 0.568 | 0.500 | 0.500 |
| Random Forest | 0.516 | 0.504 | 0.505 |
| XGBoost | **0.512** | **0.526** | 0.535 |
| Baseline ("always up") | 0.568 | 0.500 | — |

XGBoost is the only model showing any signal above chance, and it is weak
(ROC-AUC 0.526). Both other models are indistinguishable from random guessing.
Note that logistic regression *matches* the naive baseline on accuracy while
having ROC-AUC of exactly 0.500 — it degenerated into predicting a single class.

## Data

- **Source:** `^GSPC` (S&P 500 index) via `yfinance`, `auto_adjust=True`
- **Full range:** 2021-01-04 to 2026-06-30 (1,378 trading days)
- **Modelling set:** 2021-07-01 to 2026-06-29 (1,253 rows)

The first six months are loaded as a **warm-up buffer only**, so that rolling
indicators (MA-49, MACD, RSI-14) are fully defined on the first day of the
training window rather than being NaN or computed from a partial window.

**Target:** `1` if the next day's close-to-close return is positive, `0` otherwise.
Growth days make up 53.96% of the full sample — 52.9% in train, 56.8% in test.
That imbalance is why accuracy alone is misleading here and why the "always up"
baseline is reported alongside every model.

## Features (16)

| Group | Features |
|---|---|
| Lagged returns | daily returns over the previous 1–6 days |
| Moving averages | close price relative to its 7, 14, 21, 49-day MA |
| Momentum | RSI-14 (Wilder's smoothing) |
| Trend | MACD line, signal line, histogram (price-normalised) |
| Volatility | rolling std of daily returns over 5 and 21 days |

## Validation design

This is the part that matters most, and the part most often done wrong in
this kind of project.

- **Hold-out split by time, not at random.** Train: 1,003 days
  (2021-07-01 – 2025-06-30). Test: the final 250 days
  (2025-07-01 – 2026-06-29). The test year is never touched during tuning.
- **`TimeSeriesSplit(n_splits=5)`** for hyperparameter search. Ordinary
  K-fold cross-validation would train on future data and score on past data,
  producing optimistically biased estimates that do not survive out of sample.
- **No shuffling anywhere**, and features at day *t* use only information
  available at the close of day *t*.
- **Scoring on ROC-AUC**, not accuracy, so that a model cannot win by
  predicting the majority class.

Hyperparameter search: `GridSearchCV` for logistic regression (L1/L2, C over
0.001–10, with and without class weighting) and random forest (shallow depths
2–5, `min_samples_leaf` 5–20 — regularisation aimed directly at noise);
`RandomizedSearchCV` for XGBoost, since a full grid over its parameter space
would be prohibitively expensive.

## Results and interpretation

**1. No overfitting.** CV and test scores are nearly identical for every model
(e.g. random forest: 0.505 CV vs 0.504 test). The models are not memorising
noise — there simply is very little to learn.

**2. Logistic regression collapsed to a constant.** The best configuration
selected `penalty='l1'` with `C=0.001`, which drove **every coefficient to
exactly zero**. The model reduced to predicting the majority class, which is
why its accuracy equals the baseline and its ROC-AUC is exactly 0.500. This is
not a bug — it is the most informative result in the project: with L1 free to
choose, it kept nothing. There is no usable *linear* signal in these features.

**3. Feature importances are flat where they matter.** XGBoost spreads
importance almost uniformly across all 16 features (roughly 0.04–0.10 each)
with no dominant predictor. Random forest does concentrate on a few features
(`ret_lag_5`, `ret_lag_4`, `close_to_ma_49`), but since that model performs at
chance level, those rankings reflect noise rather than signal.

**4. This is what the efficient-market hypothesis predicts.** In its weak form,
past prices should not help predict future prices. A well-executed experiment
on daily data with standard technical indicators reproducing exactly that is
the expected outcome, not a failure of method.

## Caveats and next steps

Being explicit about what this project does *not* establish:

- **It rules out one specific setup**, not predictability in general: daily
  horizon, one index, technical indicators derived from its own price history.
- **No transaction costs were modelled.** They would only make a marginal
  edge worse, so the conclusion stands, but any claim about a tradable
  strategy would require them.
- **Test window is a single year (250 days).** Differences of 0.02 in ROC-AUC
  on that sample size are within noise; the honest reading of the table is
  "nothing works", not "XGBoost works slightly".
- **Naming nuance:** `ret_lag_1` is implemented as the current day's return
  (`shift(0)`), which is known at the close when the prediction is made — so
  there is no look-ahead, but the indices are offset by one relative to
  what the names suggest.
- **Where a signal would more plausibly live:** longer horizons, cross-asset
  and macro features, order-flow or volatility data, and regime-aware models —
  rather than more complex models on the same 1,250 observations, which would
  buy overfitting rather than signal.

## Stack

Python, pandas, NumPy, scikit-learn, XGBoost, yfinance, matplotlib

## Repository

```
├── notebook.ipynb    # full analysis, top to bottom
└── README.md
```

The notebook downloads data at runtime via `yfinance`, so it is reproducible
without bundled data files. Results may shift slightly as the date range rolls
forward.
