# Claims Model Validation Report

Data: freMTPL2 (French motor third-party liability), 678,013 policies, 36,056 claims, 358,360 policy-years. Test set: 135,603 policies (seed 42).

## Frequency

| Model | Held-out Poisson deviance | Gini |
|---|---|---|
| Null | 0.62738 | - |
| Poisson GLM | 0.59692 | 0.298 |
| Gradient boosting | 0.57759 | 0.346 |

Deviance improvement over GLM: **3.24%**.

## Severity

Gamma deviance: null 1.3735, GLM 1.3981, GBM 1.3694. Severity is the weak link (few, heavy-tailed claims).

## Loss-ratio bootstrap intervals

Premium proxy = GLM pure premium / 70% (dataset has no premiums). Observed loss ratio fell inside the 95% interval in **8 of 10** segments.

| Segment | Predicted LR | Observed LR | 95% interval | Covered |
|---|---|---|---|---|
| 1 | 48.3% | 74.3% | 35.1% - 67.0% | no |
| 2 | 57.9% | 70.0% | 43.1% - 76.5% | yes |
| 3 | 62.3% | 76.6% | 46.6% - 81.2% | yes |
| 4 | 65.0% | 68.8% | 50.6% - 81.6% | yes |
| 5 | 68.0% | 57.0% | 53.5% - 87.2% | yes |
| 6 | 70.4% | 62.0% | 57.1% - 87.9% | yes |
| 7 | 68.7% | 72.3% | 55.1% - 88.2% | yes |
| 8 | 65.2% | 80.6% | 52.3% - 79.5% | no |
| 9 | 60.7% | 71.0% | 50.1% - 73.9% | yes |
| 10 | 89.4% | 88.9% | 78.9% - 100.3% | yes |

## Limitations (actual / expected claims, GBM)

| Slice | Claims | A/E GLM | A/E GBM |
|---|---|---|---|
| Drivers under 25 | 479 | 1.07 | 1.01 |
| Drivers 70+ | 556 | 0.95 | 0.93 |
| Malus (BonusMalus > 100) | 238 | 0.94 | 0.95 |
| BonusMalus > 120 | 87 | 0.94 | 0.96 |
| Dense urban (> 10,000/km2) | 230 | 1.00 | 1.00 |
| Diesel, power >= 9 | 407 | 1.01 | 1.01 |
| Vehicles 0-1 years | 1,569 | 0.97 | 0.97 |
| Vehicles 15+ years | 628 | 1.01 | 1.06 |
| Top 5% predicted risk | 1,120 | 1.63 | 0.98 |
| Bottom 50% predicted risk | 2,423 | 0.87 | 1.01 |

Other caveats: proxy premiums; claim amounts capped at 100,000; severity rescaled because only ~74% of claims carry an amount; single time snapshot (no drift testing); sparse high-risk slices are noisy.
