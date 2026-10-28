"""Write reports/report.md from outputs/results.json.  Run: python -m src.report"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    r = json.loads((ROOT / "outputs" / "results.json").read_text())
    m, fr, cv = r["meta"], r["frequency"], r["coverage"]
    L = [f"# Claims Model Validation Report", "",
         f"Data: {m['dataset']}, {m['policies']:,} policies, {m['claims']:,} claims, {m['exposure_years']:,.0f} policy-years. "
         f"Test set: {m['test_policies']:,} policies (seed {m['seed']}).", "",
         "## Frequency", "",
         "| Model | Held-out Poisson deviance | Gini |", "|---|---|---|",
         f"| Null | {fr['null_deviance']:.5f} | - |",
         f"| Poisson GLM | {fr['glm_deviance']:.5f} | {fr['glm_gini']:.3f} |",
         f"| Gradient boosting | {fr['gbm_deviance']:.5f} | {fr['gbm_gini']:.3f} |", "",
         f"Deviance improvement over GLM: **{fr['improvement_pct']:.2f}%**.", "",
         "## Severity", "",
         f"Gamma deviance: null {r['severity']['null_deviance']:.4f}, GLM {r['severity']['glm_deviance']:.4f}, "
         f"GBM {r['severity']['gbm_deviance']:.4f}. Severity is the weak link (few, heavy-tailed claims).", "",
         "## Loss-ratio bootstrap intervals", "",
         f"Premium proxy = GLM pure premium / {m['target_lr']:.0%} (dataset has no premiums). "
         f"Observed loss ratio fell inside the 95% interval in **{cv['covered']} of {cv['total']}** segments.", "",
         "| Segment | Predicted LR | Observed LR | 95% interval | Covered |", "|---|---|---|---|---|"]
    for s in r["loss_ratio_segments"]:
        L.append(f"| {s['segment']} | {s['predicted_lr']:.1%} | {s['observed_lr']:.1%} | {s['ci_low']:.1%} - {s['ci_high']:.1%} | {'yes' if s['covered'] else 'no'} |")
    L += ["", "## Limitations (actual / expected claims, GBM)", "", "| Slice | Claims | A/E GLM | A/E GBM |", "|---|---|---|---|"]
    for s in r["slices"]:
        L.append(f"| {s['slice']} | {s['claims']:,} | {s['ae_glm']:.2f} | {s['ae_gbm']:.2f} |")
    L += ["", "Other caveats: proxy premiums; claim amounts capped at 100,000; severity rescaled because only ~74% of claims carry an amount; "
          "single time snapshot (no drift testing); sparse high-risk slices are noisy.", ""]
    out = ROOT / "reports" / "report.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(L))
    print("wrote", out)


if __name__ == "__main__":
    main()
