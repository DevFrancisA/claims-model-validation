# Auto Insurance Claims Model Validation

Frequency and severity pricing models on 678,013 real French motor policies (freMTPL2), validated with
lift charts, calibration, bootstrap loss-ratio intervals and slice analysis. Runs fully locally.

## Quick start
```bash
pip install -r requirements.txt
python fetch_data.py        # downloads freMTPL2 from OpenML (~10 s)
python -m src.pipeline      # SQL load + clean, train, validate, write outputs/ and reports/report.md (~1-2 min)
python app.py               # local UI at http://localhost:8000
```

## What's inside
- `sql/clean.sql` - SQLite cleaning/joining of the raw tables
- `src/features.py` - driver, vehicle and region feature engineering; GLM banding
- `src/pipeline.py` - Poisson GLM baseline vs gradient-boosted frequency model, Gamma severity, validation
- `app.py` + `ui/index.html` - local dashboard: metrics, lift/calibration, loss-ratio intervals, limitations,
  a what-if policy scorer and a 10-slide business summary (arrow keys to navigate)
- `reports/report.md` - generated written report; `r/clean_check.R` - optional R cross-check of the cleaning step

## Method notes
- 80/20 train/test split (seed 42); GBM tuned on a validation slice of train only; final GBM is a 4-seed average.
- Frequency: Poisson loss, exposure as weight. Severity: Gamma on claim-level averages, capped at 100,000.
- freMTPL2 has **no premium column**. Loss ratios use a proxy premium (GLM pure premium / 70%). Severity is rescaled
  so train expected loss matches train actual loss, because only ~74% of recorded claims carry an amount.
- Bootstrap intervals resample policies, simulate Poisson counts, and draw heavy-tailed severities from empirical residuals.

Headline numbers are in `outputs/results.json` after you run the pipeline; the UI reads from it.
