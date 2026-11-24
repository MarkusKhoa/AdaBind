#!/usr/bin/env python
"""Threshold-dependent metrics (precision, recall, F1, MCC) of P2Rank against AdaBind, small-molecule label.

AUPRC and AUROC (scripts/p2rank_baseline.py) do not depend on a decision threshold; F1 and MCC do, so every
threshold used here is written to reports/p2rank/p2rank_small_thresholds.csv.

Thresholds
  AdaBind          0.4901, fixed on the DevSet1010 out-of-fold predictions (MCC optimum) and read from
                   reports/finalize_results/manifest.json; it is not recomputed here.
  P2Rank (Dev)     MCC optimum of P2Rank's residue probability on DevSet1010, with the same exact search the
                   project uses for its own models (pbind.evaluation.thresholds.optimize_thresholds_exact).
                   This is the primary P2Rank row: no test label enters the choice.
  P2Rank (0.5)     the 0.5 default that the published baselines (bindNode24, bindEmbed21DL) are scored at.
  P2Rank (test)    threshold tuned on the test split itself. It is an upper bound that favours P2Rank, and is shown
                   only to check that the comparison does not hinge on how the threshold was chosen.

Metric definitions are those of the manuscript (scripts/finalize_metrics.py): per-protein value, mean over the proteins
with TP+FP+FN > 0, protein-level bootstrap (B = 10,000, seed 2026), SD and percentile 95% interval. The paired
difference AdaBind minus P2Rank uses the same resamples for both methods.

Run P2Rank on the test structures (out_alphafold) and on DevSet1010 (out_alphafold_dev) first; both with the
``alphafold`` configuration and a space-free path list (see scripts/p2rank_baseline.py).

    python scripts/p2rank_threshold_metrics.py --task small|metal [--boot 10000] [--seed 2026]

Tasks: ``small`` (MCC threshold criterion, as for the small-molecule models) and ``metal`` (F1 criterion, as for the
metal model, see the threshold paragraph of the manuscript).
"""
import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import finalize_metrics as FM  # noqa: E402  (score, protein_counts, protein_means: the manuscript's definitions)
from p2rank_baseline import load_p2rank  # noqa: E402

DEV_LABELS = ROOT / "data/zenodo_adabind_data_v1/labels/devset1010_binding_sites.csv"
PRED = ROOT / "reports/finalize_results/predictions/test_residue_predictions.csv.gz"
MANIFEST = ROOT / "reports/finalize_results/manifest.json"
PUBLISHED = ROOT / "reports/finalize_results/metrics/table34_mean_sd.csv"
DIFF_METRICS = ("precision", "recall", "F1", "MCC")
# per task: Dev label column, test label column, residue-probability column prefix, the criterion the project used
# for that task's own threshold, and the key of its threshold in manifest.json
TASKS = {
    "small": {"dev": "small_binding", "label": "y_small_binding", "prefix": "p_small",
               "criterion": "dev_oof_mcc_opt", "manifest": "small", "table_task": "small"},
    "metal": {"dev": "metal_binding", "label": "y_metal_binding", "prefix": "p_metal",
               "criterion": "dev_oof_positive_f1_opt", "manifest": "metal", "table_task": "metal"},
}


def thresholds_module():
    """Load pbind/evaluation/thresholds.py by path (it needs only numpy and sklearn; the package __init__ needs torch)."""
    spec = importlib.util.spec_from_file_location("pbind_thresholds", ROOT / "src/pbind/evaluation/thresholds.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dev_frame(dev_dir: Path, col: str) -> pd.DataFrame:
    """One row per DevSet1010 residue of the proteins P2Rank was run on (resid is 1-based, as in P2Rank)."""
    have = {p.name[: -len(".pdb_residues.csv")] for p in dev_dir.glob("*.pdb_residues.csv")}
    parts = []
    for r in pd.read_csv(DEV_LABELS).itertuples():
        pid = str(r.prot_id)
        if pid in have:
            y = ast.literal_eval(getattr(r, col))
            parts.append(pd.DataFrame({"prot_id": pid, "resid": np.arange(1, len(y) + 1), "y": y}))
    return pd.concat(parts, ignore_index=True)


def pooled_counts(y, p, t):
    pred = p >= t
    tp, fp = int(np.sum(pred & (y == 1))), int(np.sum(pred & (y == 0)))
    return {"n_residue": int(len(y)), "n_positive": int(np.sum(y == 1)), "n_predicted_positive": int(pred.sum()),
            "tp": tp, "fp": fp, "fn": int(np.sum(~pred & (y == 1)))}


def paired_difference(y, prot, a, b, boot, seed):
    """AdaBind minus P2Rank for the per-protein metrics, on shared bootstrap resamples (same rule as FM.score)."""
    n = int(prot.max()) + 1
    y = np.asarray(y, float)
    ca = FM.protein_counts(y, a[0] >= a[1], prot, n)
    cb = FM.protein_counts(y, b[0] >= b[1], prot, n)
    one = np.ones(n)
    pa, pb = FM.protein_means(ca, one), FM.protein_means(cb, one)
    reps = {k: [] for k in DIFF_METRICS}
    rng = np.random.default_rng(seed)
    for _ in range(boot):
        cnt = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        if np.sum(cnt[prot] * y) == 0:
            continue
        ma, mb = FM.protein_means(ca, cnt), FM.protein_means(cb, cnt)
        for k in DIFF_METRICS:
            reps[k].append(ma[k] - mb[k])
    rows = []
    for k in DIFF_METRICS:
        r = np.asarray(reps[k], float)
        rows.append({"metric": k, "diff_AdaBind_minus": pa[k] - pb[k], "sd": float(np.nanstd(r, ddof=1)),
                     "ci_lo": float(np.nanpercentile(r, 2.5)), "ci_hi": float(np.nanpercentile(r, 97.5))})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=sorted(TASKS), default="small")
    ap.add_argument("--p2rank-test", default=str(ROOT / "data/external/p2rank/out_alphafold"))
    ap.add_argument("--p2rank-dev", default=str(ROOT / "data/external/p2rank/out_alphafold_dev"))
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    cfg = TASKS[args.task]
    opt = thresholds_module().optimize_thresholds_exact
    crit = cfg["criterion"]

    # ---- P2Rank threshold chosen on DevSet1010 only, with the criterion this task's own model uses
    dev = dev_frame(Path(args.p2rank_dev), cfg["dev"])
    dev["p"] = load_p2rank(Path(args.p2rank_dev), dev)
    t_dev = opt(dev.y.to_numpy(), dev.p.to_numpy())[crit]
    print(f"[{args.task}] DevSet1010: {dev.prot_id.nunique()} proteins, {len(dev)} residues, prevalence "
          f"{dev.y.mean():.3f}; P2Rank threshold ({crit}) {t_dev:.4f}")

    t_ada = float(json.loads(MANIFEST.read_text())["thresholds"][cfg["manifest"]])
    pred = pd.read_csv(PRED)
    pred["prot_id"] = pred.prot_id.astype(str)
    pref = cfg["prefix"]
    pred[f"{pref}__P2Rank"] = load_p2rank(Path(args.p2rank_test), pred)
    published = pd.read_csv(PUBLISHED)

    metric_rows, diff_rows, thr_rows = [], [], []
    for split, d in pred.groupby("split"):
        d = d.reset_index(drop=True)
        y = d[cfg["label"]].to_numpy()
        prot = pd.factorize(d.prot_id)[0]
        ada, p2 = d[f"{pref}__AdaBind"].to_numpy(), d[f"{pref}__P2Rank"].to_numpy()
        t_test = opt(y, p2)[crit]  # exact optimum on this split (upper bound for P2Rank)
        crit_name = "MCC" if crit.endswith("mcc_opt") else "F1"
        variants = {
            "AdaBind": (ada, t_ada, f"DevSet1010 out-of-fold, {crit_name} optimum (manifest.json)"),
            "P2Rank (Dev threshold)": (p2, t_dev, f"DevSet1010, exact {crit_name} optimum"),
            "P2Rank (0.5)": (p2, 0.5, "fixed 0.5, as for the published baselines"),
            "P2Rank (test-tuned, upper bound)": (p2, t_test, f"{split} itself, exact {crit_name} optimum"),
        }
        models = {m: (p, t) for m, (p, t, _) in variants.items()}
        for row in FM.score(y, prot, models, args.boot, args.seed):
            metric_rows.append({"split": split, **row})
        for m, (p, t, src) in variants.items():
            thr_rows.append({"split": split, "method": m, "threshold": t, "threshold_source": src,
                             **pooled_counts(y, p, t)})
        for m in ("P2Rank (Dev threshold)", "P2Rank (0.5)", "P2Rank (test-tuned, upper bound)"):
            for row in paired_difference(y, prot, models["AdaBind"], models[m], args.boot, args.seed):
                diff_rows.append({"split": split, "comparator": m, **row})

    metrics = pd.DataFrame(metric_rows)
    # consistency gate: AdaBind's rows must equal the published table (same code path as the manuscript)
    for r in metrics[metrics.method == "AdaBind"].itertuples():
        ref = published[(published.task == cfg["table_task"]) & (published.split == r.split)
                        & (published.method == "ours") & (published.metric == r.metric)].value.iloc[0]
        assert abs(ref - r.value) < 1e-9, (r.split, r.metric, ref, r.value)
    print(f"[{args.task}] AdaBind AUPRC/AUROC/precision/recall/F1/MCC equal table34_mean_sd.csv (consistency gate passed)")

    dest = ROOT / "reports" / "p2rank"
    dest.mkdir(parents=True, exist_ok=True)
    tag = args.task
    metrics.to_csv(dest / f"p2rank_{tag}_threshold_metrics.csv", index=False)
    pd.DataFrame(diff_rows).to_csv(dest / f"p2rank_{tag}_threshold_diff.csv", index=False)
    pd.DataFrame(thr_rows).to_csv(dest / f"p2rank_{tag}_thresholds.csv", index=False)

    pd.set_option("display.width", 220)
    show = metrics[metrics.metric.isin(["AUPRC", "AUROC", "precision", "recall", "F1", "MCC"])]
    print("\nThresholds and pooled counts")
    print(pd.DataFrame(thr_rows).round(4).to_string(index=False))
    print("\nPer-protein metrics (SD and 95% CI from the protein bootstrap)")
    print(show[["split", "method", "metric", "value", "sd", "ci_lo", "ci_hi", "threshold", "n_protein_included"]]
          .round(3).to_string(index=False))
    print("\nPaired difference AdaBind minus P2Rank")
    print(pd.DataFrame(diff_rows).round(3).to_string(index=False))

if __name__ == "__main__":
    main()
