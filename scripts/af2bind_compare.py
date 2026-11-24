#!/usr/bin/env python
"""AF2BIND against AdaBind and P2Rank, small-molecule label, on TestSet300 and TestSetNew46.

Requires the AF2BIND predictions written by scripts/af2bind_infer.py (data/external/af2bind/out/<prot_id>.csv).
Metric definitions, bootstrap (B = 10,000, seed 2026, protein-level, shared resamples) and paired differences are those
of scripts/finalize_metrics.py and scripts/p2rank_threshold_metrics.py.

Thresholds (fixed before this comparison, none tuned on test):
  AdaBind          0.4901, DevSet1010 out-of-fold MCC optimum (reports/finalize_results/manifest.json)
  AF2BIND          the AdaBind threshold, so the F1/MCC columns use one public threshold for both models
  P2Rank           0.4586, DevSet1010 MCC optimum (reports/p2rank/p2rank_small_thresholds.csv)

Outputs: reports/p2rank/af2bind_small_metrics.csv (values, SD, percentile 95% intervals) and
reports/p2rank/af2bind_small_diff.csv (paired differences AF2BIND minus AdaBind and minus P2Rank).

    conda run -n AdaBind python scripts/af2bind_compare.py [--boot 10000] [--seed 2026]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import finalize_metrics as FM  # noqa: E402
from p2rank_baseline import load_p2rank  # noqa: E402
from p2rank_threshold_metrics import paired_difference  # noqa: E402

PRED = ROOT / "reports/finalize_results/predictions/test_residue_predictions.csv.gz"
AF_OUT = ROOT / "data/external/af2bind/out"
P2_TEST = ROOT / "data/external/p2rank/out_alphafold"
THR = ROOT / "reports/p2rank/p2rank_small_thresholds.csv"
MANIFEST = ROOT / "reports/finalize_results/manifest.json"
PUBLISHED = ROOT / "reports/finalize_results/metrics/table34_mean_sd.csv"


def af2bind_column(pred: pd.DataFrame) -> pd.Series:
    parts = []
    for pid, g in pred.groupby("prot_id", sort=False):
        a = pd.read_csv(AF_OUT / f"{pid}.csv")
        m = dict(zip(a.resid.astype(int), a.p_bind.astype(float)))
        parts.append(pd.Series([m[int(r)] for r in g.resid], index=g.index))
    return pd.concat(parts).sort_index()


def paired_curve(y, prot, a, b, boot, seed):
    """AUPRC and AUROC of a minus b (pooled residues) on the same protein resamples as FM.score."""
    n = int(prot.max()) + 1
    y = np.asarray(y, float)
    pa, pb = FM.prep(a), FM.prep(b)
    one = np.ones(n)
    ma, mb = FM.curve_metrics(y, one[prot], pa), FM.curve_metrics(y, one[prot], pb)
    pt = {"AUPRC": ma[0] - mb[0], "AUROC": ma[1] - mb[1]}
    reps = {"AUPRC": [], "AUROC": []}
    rng = np.random.default_rng(seed)
    for _ in range(boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)[prot]
        if np.sum(w * y) == 0:
            continue
        xa, ra = FM.curve_metrics(y, w, pa)
        xb, rb = FM.curve_metrics(y, w, pb)
        reps["AUPRC"].append(xa - xb)
        reps["AUROC"].append(ra - rb)
    rows = []
    for k in ("AUPRC", "AUROC"):
        r = np.asarray(reps[k], float)
        rows.append({"metric": k, "diff": pt[k], "sd": float(np.nanstd(r, ddof=1)),
                     "ci_lo": float(np.nanpercentile(r, 2.5)), "ci_hi": float(np.nanpercentile(r, 97.5))})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    t_ada = float(json.loads(MANIFEST.read_text())["thresholds"]["small"])
    thr = pd.read_csv(THR)
    t_p2 = float(thr[(thr.split == "TestSet300") & (thr.method == "P2Rank (Dev threshold)")].threshold.iloc[0])

    pred = pd.read_csv(PRED)
    pred["prot_id"] = pred.prot_id.astype(str)
    pred["p_af"] = af2bind_column(pred)
    pred["p_p2"] = load_p2rank(P2_TEST, pred)
    pred["p_ada"] = pred["p_small__AdaBind"]
    published = pd.read_csv(PUBLISHED)

    metric_rows, diff_rows = [], []
    for split, d in pred.groupby("split"):
        d = d.reset_index(drop=True)
        y = d.y_small_binding.to_numpy()
        prot = pd.factorize(d.prot_id)[0]
        af, ada, p2 = d.p_af.to_numpy(), d.p_ada.to_numpy(), d.p_p2.to_numpy()
        models = {"AF2BIND": (af, t_ada), "AdaBind": (ada, t_ada), "P2Rank": (p2, t_p2)}
        rows = FM.score(y, prot, models, args.boot, args.seed)
        metric_rows += [{"split": split, **r} for r in rows]
        for name, comp in (("AdaBind", (ada, t_ada)), ("P2Rank", (p2, t_p2))):
            for r in paired_curve(y, prot, af, comp[0], args.boot, args.seed):
                diff_rows.append({"split": split, "comparison": f"AF2BIND - {name}", **r})
            for r in paired_difference(y, prot, (af, t_ada), comp, args.boot, args.seed):
                diff_rows.append({"split": split, "comparison": f"AF2BIND - {name}", **r})

    metrics = pd.DataFrame(metric_rows)
    # consistency gate: AdaBind's values must equal the published table (same code path as the manuscript)
    for r in metrics[metrics.method == "AdaBind"].itertuples():
        ref = published[(published.task == "small") & (published.split == r.split)
                        & (published.method == "ours") & (published.metric == r.metric)].value.iloc[0]
        assert abs(ref - r.value) < 1e-9, (r.split, r.metric, ref, r.value)
    print("AdaBind values equal table34_mean_sd.csv (consistency gate passed)")

    dest = ROOT / "reports" / "p2rank"
    dest.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(dest / "af2bind_small_metrics.csv", index=False)
    diffs = pd.DataFrame(diff_rows)
    diffs.to_csv(dest / "af2bind_small_diff.csv", index=False)

    pd.set_option("display.width", 200)
    print(metrics[["split", "method", "metric", "value", "sd", "ci_lo", "ci_hi", "threshold"]].round(3).to_string(index=False))
    print()
    print(diffs.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
