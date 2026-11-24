#!/usr/bin/env python
"""Where does P2Rank lose metal-binding residues? TestSet300 is the primary set; TestSetNew46 is reported only.

A metal site is a few donor atoms around a point, whereas P2Rank predicts ligandable pockets. If P2Rank sees metal sites
only where they line a pocket, metal residues that also bind a small molecule ("co-localised") should fall inside its
predicted pockets far more often than metal-only residues.

Residue groups: all residues, small-molecule residues, metal co-localised (metal and small-molecule label), metal-only
(metal, no small-molecule label), and metal-only split by residue type (Cys/His, Asp/Glu, other).
Measures per group: fraction inside a P2Rank pocket (``pocket`` != 0 in its *_residues.csv); fraction detected by P2Rank
at its metal threshold chosen on DevSet1010 (reports/p2rank/p2rank_metal_thresholds.csv); fraction detected by AdaBind at
its metal threshold (reports/finalize_results/manifest.json). Protein bootstrap (B = 10,000, seed 2026) for every
fraction and for the difference co-localised minus metal-only, on the same resamples.

Pre-declared reading: if metal-only residues are inside P2Rank pockets (or detected by P2Rank) clearly less often than
co-localised ones (95% CI of the difference excludes zero), the text may say that P2Rank recovers metal residues mainly
where they line a small-molecule pocket; otherwise only the measured fractions are reported.

    python scripts/p2rank_metal_failure.py [--boot 10000] [--seed 2026]

Output: reports/p2rank/p2rank_metal_failure.csv and reports/p2rank/p2rank_metal_failure_diff.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "reports/finalize_results/predictions/test_residue_predictions.csv.gz"
MANIFEST = ROOT / "reports/finalize_results/manifest.json"
P2RANK_THRESHOLDS = ROOT / "reports/p2rank/p2rank_metal_thresholds.csv"


def load_p2rank_columns(out_dir: Path, pred: pd.DataFrame) -> pd.DataFrame:
    """P2Rank residue probability and pocket membership for every row of ``pred`` (matched by chain-A position)."""
    parts = []
    for pid, g in pred.groupby("prot_id", sort=False):
        r = pd.read_csv(out_dir / f"{pid}.pdb_residues.csv", skipinitialspace=True)
        r.columns = [c.strip() for c in r.columns]
        r = r[r.chain.astype(str).str.strip() == "A"] if "A" in set(r.chain.astype(str).str.strip()) else r
        prob = dict(zip(r.residue_label.astype(int), r.probability.astype(float)))
        pocket = dict(zip(r.residue_label.astype(int), r.pocket.astype(int)))
        parts.append(pd.DataFrame({"p2_prob": [prob[int(i)] for i in g.resid],
                                   "p2_in_pocket": [pocket[int(i)] != 0 for i in g.resid]}, index=g.index))
    return pd.concat(parts).sort_index()


def fraction(num, den, cnt):
    d = cnt @ den
    return (cnt @ num) / d if d > 0 else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p2rank-out", default=str(ROOT / "data/external/p2rank/out_alphafold"))
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    thr = pd.read_csv(P2RANK_THRESHOLDS)
    t_p2 = float(thr[thr.method == "P2Rank (Dev threshold)"].threshold.iloc[0])
    t_ada = float(json.loads(MANIFEST.read_text())["thresholds"]["metal"])
    print(f"thresholds: P2Rank {t_p2:.4f} (DevSet1010, F1 optimum), AdaBind {t_ada:.4f} (Dev out-of-fold)")

    pred = pd.read_csv(PRED)
    pred["prot_id"] = pred.prot_id.astype(str)
    pred = pred.join(load_p2rank_columns(Path(args.p2rank_out), pred))

    rows, diffs = [], []
    for split, d in pred.groupby("split"):
        d = d.reset_index(drop=True)
        prot = pd.factorize(d.prot_id)[0]
        n = int(prot.max()) + 1
        metal, small = d.y_metal_binding.to_numpy() == 1, d.y_small_binding.to_numpy() == 1
        aa = d.aa.astype(str).to_numpy()
        metal_only = metal & ~small
        groups = {
            "all residues": np.ones(len(d), bool),
            "small-molecule residues": small,
            "metal residues (all)": metal,
            "metal co-localised (metal and small)": metal & small,
            "metal-only": metal_only,
            "metal-only Cys/His": metal_only & np.isin(aa, ["C", "H"]),
            "metal-only Asp/Glu": metal_only & np.isin(aa, ["D", "E"]),
            "metal-only other": metal_only & ~np.isin(aa, ["C", "H", "D", "E"]),
        }
        measures = {
            "inside a P2Rank pocket": d.p2_in_pocket.to_numpy(),
            "detected by P2Rank": d.p2_prob.to_numpy() >= t_p2,
            "detected by AdaBind": d.p_metal__AdaBind.to_numpy() >= t_ada,
        }
        # per-protein numerators and denominators, so that a resample only reweights proteins
        den = {g: np.bincount(prot, weights=m.astype(float), minlength=n) for g, m in groups.items()}
        num = {(g, k): np.bincount(prot, weights=(m & x).astype(float), minlength=n)
               for g, m in groups.items() for k, x in measures.items()}
        one = np.ones(n)
        reps = {key: np.full(args.boot, np.nan) for key in num}
        rng = np.random.default_rng(args.seed)
        for b in range(args.boot):
            cnt = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
            for (g, k) in num:
                reps[(g, k)][b] = fraction(num[(g, k)], den[g], cnt)
        for (g, k), r in reps.items():
            rows.append({"split": split, "group": g, "measure": k, "value": fraction(num[(g, k)], den[g], one),
                         "sd": float(np.nanstd(r, ddof=1)), "ci_lo": float(np.nanpercentile(r, 2.5)),
                         "ci_hi": float(np.nanpercentile(r, 97.5)), "n_residues": int(den[g].sum()),
                         "n_proteins": int(np.sum(den[g] > 0))})
        for k in measures:
            a, c = ("metal co-localised (metal and small)", k), ("metal-only", k)
            r = reps[a] - reps[c]
            diffs.append({"split": split, "measure": k, "co_localised": fraction(num[a], den[a[0]], one),
                          "metal_only": fraction(num[c], den[c[0]], one),
                          "difference": fraction(num[a], den[a[0]], one) - fraction(num[c], den[c[0]], one),
                          "sd": float(np.nanstd(r, ddof=1)), "ci_lo": float(np.nanpercentile(r, 2.5)),
                          "ci_hi": float(np.nanpercentile(r, 97.5))})

    dest = ROOT / "reports" / "p2rank"
    dest.mkdir(parents=True, exist_ok=True)
    res, dif = pd.DataFrame(rows), pd.DataFrame(diffs)
    res.to_csv(dest / "p2rank_metal_failure.csv", index=False)
    dif.to_csv(dest / "p2rank_metal_failure_diff.csv", index=False)
    pd.set_option("display.width", 220)
    print(res.round(3).to_string(index=False))
    print("\nCo-localised minus metal-only (same resamples)")
    print(dif.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
