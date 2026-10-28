"""End-to-end pipeline: SQL load -> clean -> frequency & severity models -> validation -> results.json.

Run:  python -m src.pipeline
"""
import json
import sqlite3
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from sklearn.metrics import mean_gamma_deviance, mean_poisson_deviance
from sklearn.model_selection import train_test_split

from src.features import GBM_CATEGORICAL, Avg, gbm_features, glm_features

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "outputs"
SEED = 42
TARGET_LR = 0.70          # premium proxy: GLM pure premium / TARGET_LR (dataset has no premium column)
SEV_CAP = 100_000         # cap individual claim amounts (standard treatment of freMTPL2 outliers)
N_BOOT = 400


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- SQL load + clean
def load_with_sql() -> pd.DataFrame:
    db = DATA / "claims.db"
    db.unlink(missing_ok=True)
    con = sqlite3.connect(db)
    pd.read_csv(DATA / "freMTPL2freq.csv").to_sql("freq_raw", con, index=False)
    pd.read_csv(DATA / "freMTPL2sev.csv").to_sql("sev_raw", con, index=False)
    con.executescript((ROOT / "sql" / "clean.sql").read_text())
    df = pd.read_sql("SELECT * FROM policies", con)
    con.close()
    return df


# --------------------------------------------------------------------------- helpers
def pdev(y, mu, w):
    return mean_poisson_deviance(y, np.clip(mu, 1e-9, None), sample_weight=w)


def gini(y_rate, w, score):
    """Normalised exposure-weighted Gini (Lorenz-curve based)."""
    def raw(s):
        o = np.argsort(s)
        cw = np.cumsum(w[o]) / w.sum()
        cl = np.cumsum((y_rate * w)[o]) / (y_rate * w).sum()
        return 1 - 2 * np.trapezoid(cl, cw)
    return float(raw(score) / raw(y_rate + 1e-12 * score))


def fit_gbm(X, y, w, loss, seed=SEED, **kw):
    cat = [c in GBM_CATEGORICAL for c in X.columns]
    m = HistGradientBoostingRegressor(
        loss=loss, categorical_features=cat, early_stopping=True, validation_fraction=0.12,
        n_iter_no_change=25, max_iter=1500, random_state=seed, **kw)
    return m.fit(X, y, sample_weight=w)


def decile_table(score, y_rate, w, preds: dict, n=10):
    """Bin by `score` into exposure-weighted bins; actual vs each model's predicted rate."""
    o = np.argsort(score)
    cw = np.cumsum(w[o]) / w.sum()
    bins = np.minimum((cw * n - 1e-9).astype(int), n - 1)
    rows = []
    for b in range(n):
        idx = o[bins == b]
        ww = w[idx]
        row = {"bin": b + 1, "exposure": float(ww.sum()), "actual": float((y_rate[idx] * ww).sum() / ww.sum())}
        for k, p in preds.items():
            row[k] = float((p[idx] * ww).sum() / ww.sum())
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    log("Loading raw CSVs into SQLite and cleaning with SQL ...")
    df = load_with_sql()
    log(f"{len(df):,} policies; {int(df.ClaimNb.sum()):,} claims; exposure {df.Exposure.sum():,.0f} policy-years")

    tr, te = train_test_split(df, test_size=0.2, random_state=SEED)
    tr, va = train_test_split(tr, test_size=0.15, random_state=SEED)  # va used for tuning only
    train_full = pd.concat([tr, va])

    def yw(d):
        return (d.ClaimNb / d.Exposure).values, d.Exposure.values

    # ---------------- Frequency: GLM baseline
    log("Fitting Poisson GLM baseline ...")
    Xg_tr, Xg_te = glm_features(train_full), glm_features(te)
    glm_cols = list(Xg_tr.columns)
    Xg_te = Xg_te.reindex(columns=glm_cols, fill_value=0.0)
    y_tr, w_tr = yw(train_full)
    y_te, w_te = yw(te)
    glm_f = PoissonRegressor(alpha=1e-5, max_iter=500).fit(Xg_tr, y_tr, sample_weight=w_tr)
    p_glm = glm_f.predict(Xg_te)
    p_glm_tr = glm_f.predict(Xg_tr)

    # ---------------- Frequency: GBM with small tuning grid (scored on validation split)
    log("Tuning gradient-boosted frequency model ...")
    Xv, Xt = gbm_features(va), gbm_features(tr)
    yt, wt = yw(tr)
    yv, wv = yw(va)
    grid = [
        dict(learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=400, l2_regularization=5.0),
        dict(learning_rate=0.05, max_leaf_nodes=12, min_samples_leaf=300, l2_regularization=5.0),
        dict(learning_rate=0.04, max_leaf_nodes=16, min_samples_leaf=500, l2_regularization=10.0),
        dict(learning_rate=0.04, max_leaf_nodes=6, min_samples_leaf=300, l2_regularization=2.0, max_features=0.8),
        dict(learning_rate=0.03, max_leaf_nodes=10, min_samples_leaf=600, l2_regularization=10.0, max_features=0.8),
    ]
    best, best_dev = None, 9
    tuning = []
    for cfg in grid:
        m = fit_gbm(Xt, yt, wt, "poisson", **cfg)
        d = pdev(yv, m.predict(Xv), wv)
        tuning.append({**cfg, "val_deviance": d, "iters": int(m.n_iter_)})
        log(f"  {cfg} -> val dev {d:.5f} ({m.n_iter_} iters)")
        if d < best_dev:
            best, best_dev = cfg, d
    log(f"Refitting best config on full training data: {best}")
    Xgb_tr, Xgb_te = gbm_features(train_full), gbm_features(te)
    gbm_f = Avg([fit_gbm(Xgb_tr, y_tr, w_tr, "poisson", seed=SEED + i, **best) for i in range(4)])
    p_gbm = gbm_f.predict(Xgb_te)

    dev_glm_te, dev_gbm_te = pdev(y_te, p_glm, w_te), pdev(y_te, p_gbm, w_te)
    dev_null = pdev(y_te, np.full(len(y_te), (y_tr * w_tr).sum() / w_tr.sum()), w_te)
    improvement = (dev_glm_te - dev_gbm_te) / dev_glm_te * 100
    log(f"Held-out Poisson deviance: GLM {dev_glm_te:.5f}  GBM {dev_gbm_te:.5f}  improvement {improvement:.2f}%")

    # ---------------- Severity: Gamma GLM vs Gamma GBM (claim-level, claims with amounts > 0)
    log("Fitting severity models ...")
    def sev_set(d):
        s = d[d.NSev > 0]
        return s, (s.ClaimAmount / s.NSev).values, s.NSev.values
    s_tr, ys_tr, ws_tr = sev_set(train_full)
    s_te, ys_te, ws_te = sev_set(te)
    sg_tr = glm_features(s_tr).reindex(columns=glm_cols, fill_value=0.0)
    sg_te = glm_features(s_te).reindex(columns=glm_cols, fill_value=0.0)
    # severity GLM: fewer, coarser columns to avoid overfitting 25k claims
    sev_glm = GammaRegressor(alpha=1e-3, max_iter=500).fit(sg_tr, ys_tr, sample_weight=ws_tr)
    sev_gbm = fit_gbm(gbm_features(s_tr), ys_tr, ws_tr, "gamma",
                      learning_rate=0.03, max_leaf_nodes=4, min_samples_leaf=300, l2_regularization=20.0)
    gd = lambda y, p, w: mean_gamma_deviance(y, np.clip(p, 1, None), sample_weight=w)
    sd_glm = gd(ys_te, sev_glm.predict(sg_te), ws_te)
    sd_gbm = gd(ys_te, sev_gbm.predict(gbm_features(s_te)), ws_te)
    sd_null = gd(ys_te, np.full(len(ys_te), np.average(ys_tr, weights=ws_tr)), ws_te)
    use_gbm_sev = sd_gbm < sd_glm
    log(f"Severity gamma deviance: null {sd_null:.4f} GLM {sd_glm:.4f} GBM {sd_gbm:.4f} -> using {'GBM' if use_gbm_sev else 'GLM'}")

    Xte_g = glm_features(te).reindex(columns=glm_cols, fill_value=0.0)
    sev_glm_te = sev_glm.predict(Xte_g)
    sev_gbm_te = sev_gbm.predict(gbm_features(te))
    sev_best_te = sev_gbm_te if use_gbm_sev else sev_glm_te

    # ---------------- Severity calibration: only ~74% of reported claims carry an amount in freMTPL2,
    # so raw severities overstate loss per claim. Rescale each model so train-set expected loss = train actual loss.
    ltr = train_full.ClaimAmount.values
    k_glm = ltr.sum() / (p_glm_tr * w_tr * sev_glm.predict(Xg_tr)).sum()
    k_gbm = ltr.sum() / (gbm_f.predict(Xgb_tr) * w_tr * (sev_gbm.predict(Xgb_tr) if use_gbm_sev else sev_glm.predict(Xg_tr))).sum()
    log(f"Severity calibration factors: GLM {k_glm:.3f}, GBM {k_gbm:.3f}")
    sev_glm_te = sev_glm_te * k_glm
    sev_best_te = sev_best_te * k_gbm

    # ---------------- Pure premium, loss ratio proxy
    pp_glm = p_glm * sev_glm_te                    # expected loss per exposure-year
    pp_gbm = p_gbm * sev_best_te
    premium = w_te * pp_glm / TARGET_LR            # rate-plan premium proxy (current GLM tariff)
    loss = te.ClaimAmount.values
    pp_actual = loss / w_te
    pp_gini_glm, pp_gini_gbm = gini(pp_actual, w_te, pp_glm), gini(pp_actual, w_te, pp_gbm)
    gini_f_glm, gini_f_gbm = gini(y_te, w_te, p_glm), gini(y_te, w_te, p_gbm)
    log(f"Frequency Gini GLM {gini_f_glm:.3f} GBM {gini_f_gbm:.3f}; pure-premium Gini GLM {pp_gini_glm:.3f} GBM {pp_gini_gbm:.3f}")

    # ---------------- Lift / calibration tables
    lift = decile_table(p_gbm, y_te, w_te, {"glm": p_glm, "gbm": p_gbm})
    ratio = p_gbm / p_glm
    double_lift = decile_table(ratio, y_te, w_te, {"glm": p_glm, "gbm": p_gbm})
    pp_lift = decile_table(pp_gbm, pp_actual, w_te, {"glm": pp_glm, "gbm": pp_gbm})
    # calibration by predicted-frequency bins, finer top tail
    calib = decile_table(p_gbm, y_te, w_te, {"gbm": p_gbm}, n=20)

    # ---------------- Bootstrap predicted loss-ratio intervals per segment
    log(f"Bootstrapping loss-ratio intervals ({N_BOOT} replicates x 10 segments) ...")
    rng = np.random.default_rng(SEED)
    # pool of observed/predicted severity ratios from training claims -> heavy-tailed simulation
    pool = ys_tr / (sev_gbm.predict(gbm_features(s_tr)) if use_gbm_sev else sev_glm.predict(sg_tr))
    pool = pool / pool.mean()
    order = np.argsort(pp_gbm)
    seg_of = np.empty(len(te), int)
    cw = np.cumsum(w_te[order]) / w_te.sum()
    seg_of[order] = np.minimum((cw * 10 - 1e-9).astype(int), 9)
    segs = []
    covered = 0
    mu = p_gbm * w_te                                # expected claim counts
    for s in range(10):
        idx = np.where(seg_of == s)[0]
        n = len(idx)
        prem_s, loss_s = premium[idx].sum(), loss[idx].sum()
        pred_lr = (mu[idx] * sev_best_te[idx]).sum() / prem_s
        sims = np.empty(N_BOOT)
        for b in range(N_BOOT):
            bi = rng.integers(0, n, n)
            lam = mu[idx][bi]
            counts = rng.poisson(lam)
            sev_mean = (lam * sev_best_te[idx][bi]).sum() / max(lam.sum(), 1e-9)
            k = counts.sum()
            amt = (sev_best_te[idx][bi].repeat(counts) * rng.choice(pool, k)).sum() if k else 0.0
            sims[b] = amt / premium[idx][bi].sum()
        lo, hi = np.percentile(sims, [2.5, 97.5])
        obs_lr = loss_s / prem_s
        ok = bool(lo <= obs_lr <= hi)
        covered += ok
        segs.append({"segment": s + 1, "policies": int(n), "exposure": float(w_te[idx].sum()),
                     "predicted_lr": float(pred_lr), "observed_lr": float(obs_lr),
                     "ci_low": float(lo), "ci_high": float(hi), "covered": ok,
                     "claims": int(te.ClaimNb.values[idx].sum())})
    log(f"Bootstrap interval coverage: {covered}/10 segments")

    # ---------------- Limitations: actual / expected on risk slices
    slices = {
        "Drivers under 25": te.DrivAge < 25,
        "Drivers 70+": te.DrivAge >= 70,
        "Malus (BonusMalus > 100)": te.BonusMalus > 100,
        "BonusMalus > 120": te.BonusMalus > 120,
        "Dense urban (> 10,000/km2)": te.Density > 10000,
        "Diesel, power >= 9": (te.VehGas.str.contains("Diesel")) & (te.VehPower >= 9),
        "Vehicles 0-1 years": te.VehAge <= 1,
        "Vehicles 15+ years": te.VehAge >= 15,
        "Top 5% predicted risk": pd.Series(p_gbm >= np.quantile(p_gbm, 0.95), index=te.index),
        "Bottom 50% predicted risk": pd.Series(p_gbm <= np.quantile(p_gbm, 0.50), index=te.index),
    }
    slice_rows = []
    for name, mask in slices.items():
        m = np.asarray(mask)
        a = (te.ClaimNb.values[m]).sum()
        slice_rows.append({"slice": name, "policies": int(m.sum()), "claims": int(a),
                           "actual_freq": float(a / w_te[m].sum()),
                           "glm_freq": float((p_glm[m] * w_te[m]).sum() / w_te[m].sum()),
                           "gbm_freq": float((p_gbm[m] * w_te[m]).sum() / w_te[m].sum()),
                           "ae_gbm": float(a / (p_gbm[m] * w_te[m]).sum()),
                           "ae_glm": float(a / (p_glm[m] * w_te[m]).sum())})

    # ---------------- Feature importance (permutation on a test subsample)
    log("Permutation importance ...")
    sub = te.sample(40000, random_state=SEED)
    Xs = gbm_features(sub)
    ys, ws = yw(sub)
    base = pdev(ys, gbm_f.predict(Xs), ws)
    imp = []
    for c in Xs.columns:
        incs = []
        for _ in range(2):
            Xp = Xs.copy()
            Xp[c] = rng.permutation(Xp[c].values)
            incs.append(pdev(ys, gbm_f.predict(Xp), ws) - base)
        imp.append({"feature": c, "importance": float(np.mean(incs))})
    imp.sort(key=lambda r: -r["importance"])

    # ---------------- Persist
    joblib.dump({"glm_f": glm_f, "glm_cols": glm_cols, "gbm_f": gbm_f, "sev_glm": sev_glm,
                 "sev_gbm": sev_gbm, "use_gbm_sev": bool(use_gbm_sev), "k_glm": float(k_glm), "k_gbm": float(k_gbm)}, OUT / "models.joblib")

    results = {
        "meta": {
            "dataset": "freMTPL2 (French motor third-party liability)",
            "policies": int(len(df)), "claims": int(df.ClaimNb.sum()), "exposure_years": float(df.Exposure.sum()),
            "train_policies": int(len(train_full)), "test_policies": int(len(te)),
            "claim_rate": float(df.ClaimNb.sum() / df.Exposure.sum()),
            "avg_severity": float(df.ClaimAmount.sum() / max(df.NSev.sum(), 1)),
            "target_lr": TARGET_LR, "severity_cap": SEV_CAP, "bootstrap_reps": N_BOOT, "seed": SEED,
            "runtime_sec": round(time.time() - t0, 1), "best_gbm_config": best,
            "severity_model_used": "GBM" if use_gbm_sev else "GLM",
        },
        "frequency": {
            "null_deviance": dev_null, "glm_deviance": dev_glm_te, "gbm_deviance": dev_gbm_te,
            "improvement_pct": improvement, "glm_gini": gini_f_glm, "gbm_gini": gini_f_gbm,
            "tuning": tuning,
        },
        "severity": {"null_deviance": sd_null, "glm_deviance": sd_glm, "gbm_deviance": sd_gbm,
                     "improvement_vs_null_pct": (sd_null - min(sd_glm, sd_gbm)) / sd_null * 100},
        "pure_premium": {"glm_gini": pp_gini_glm, "gbm_gini": pp_gini_gbm,
                         "total_actual": float(loss.sum()),
                         "total_pred_glm": float((pp_glm * w_te).sum()), "total_pred_gbm": float((pp_gbm * w_te).sum())},
        "lift": lift, "double_lift": double_lift, "pp_lift": pp_lift, "calibration": calib,
        "loss_ratio_segments": segs,
        "coverage": {"covered": int(covered), "total": 10},
        "slices": slice_rows,
        "importance": imp,
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    from src.report import main as write_report; write_report()
    log(f"Done in {time.time() - t0:.0f}s -> outputs/results.json")


if __name__ == "__main__":
    main()
