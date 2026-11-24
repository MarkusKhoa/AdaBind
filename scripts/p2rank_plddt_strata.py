#!/usr/bin/env python
"""Does the gap between AdaBind and P2Rank depend on the confidence of the ESMFold model? (small-molecule label)

P2Rank reads the surface of the supplied model, whereas language-model predictors do not depend on its geometry. If
imperfect predicted geometry hurts P2Rank, the gap should be larger on low-confidence models. pLDDT measures model
confidence, not the apo/holo state, so this is an indirect test.

The proteins of TestSet300 and TestSetNew46 are pooled (346, as for the combined estimates of the manuscript) and split
into tertiles of the mean C-alpha pLDDT of their ESMFold model (ESMFold writes pLDDT, 0-1, in the B-factor column).
Within each tertile: residue-level AUPRC of every method (curve_metrics of scripts/finalize_metrics.py), protein
bootstrap (B = 10,000, seed 2026, same resamples for every method), and the paired gaps AdaBind - P2Rank and
bindEmbed21DL - P2Rank (bindEmbed21DL reads no structure). Interaction = gap(low tertile) - gap(high tertile), with the
tertiles resampled independently.

Pre-declared reading: an interaction > 0 whose 95% CI excludes zero supports "the gap widens on less confident models";
otherwise pLDDT is not used as evidence.

    python scripts/p2rank_plddt_strata.py [--boot 10000] [--seed 2026]

Output: reports/p2rank/p2rank_small_plddt_strata.csv and reports/p2rank/p2rank_small_plddt_interaction.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import finalize_metrics as FM  # noqa: E402  (prep, curve_metrics: the manuscript's AUPRC)
from p2rank_baseline import load_p2rank  # noqa: E402

PDB_DIR = ROOT / "data/zenodo_adabind_data_v1/structures/esmfold"
PRED = ROOT / "reports/finalize_results/predictions/test_residue_predictions.csv.gz"
BASELINE_TABLE = ROOT / "reports/p2rank/p2rank_small_auprc.csv"
METHODS = ("AdaBind", "bindNode24", "bindEmbed21DL", "P2Rank")
GAPS = (("AdaBind", "P2Rank"), ("bindEmbed21DL", "P2Rank"))


def mean_plddt(pid: str) -> float:
    with open(PDB_DIR / f"{pid}.pdb") as fh:
        vals = [float(line[60:66]) for line in fh
                if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[21] == "A"]
    return float(np.mean(vals))


def auprc_reps(y, prot, scores: dict, boot: int, seed: int):
    """Point AUPRC and bootstrap replicates for every method; a skipped resample is NaN for all methods."""
    n = int(prot.max()) + 1
    y = np.asarray(y, float)
    pre = {m: FM.prep(p) for m, p in scores.items()}
    one = np.ones(n)
    point = {m: FM.curve_metrics(y, one[prot], pre[m])[0] for m in scores}
    reps = {m: np.full(boot, np.nan) for m in scores}
    rng = np.random.default_rng(seed)
    for b in range(boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)[prot]
        if np.sum(w * y) == 0:  # same skip rule as finalize_metrics.score
            continue
        for m in scores:
            reps[m][b] = FM.curve_metrics(y, w, pre[m])[0]
    return point, reps


def spread(r):
    return {"sd": float(np.nanstd(r, ddof=1)), "ci_lo": float(np.nanpercentile(r, 2.5)),
            "ci_hi": float(np.nanpercentile(r, 97.5))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p2rank-out", default=str(ROOT / "data/external/p2rank/out_alphafold"))
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    pred = pd.read_csv(PRED)
    pred["prot_id"] = pred.prot_id.astype(str)
    pred["p_small__P2Rank"] = load_p2rank(Path(args.p2rank_out), pred)

    # consistency gate: the per-split AUPRC of every method must equal the baseline table
    base = pd.read_csv(BASELINE_TABLE)
    for split, d in pred.groupby("split"):
        y = d.y_small_binding.to_numpy(float)
        for m in METHODS:
            a = FM.curve_metrics(y, np.ones(len(y)), FM.prep(d[f"p_small__{m}"].to_numpy()))[0]
            ref = base[(base.split == split) & (base.method == m)].AUPRC.iloc[0]
            assert abs(a - ref) < 1e-9, (split, m, a, ref)
    print("per-split AUPRC equal reports/p2rank/p2rank_small_auprc.csv (consistency gate passed)")

    plddt = pd.Series({pid: mean_plddt(pid) for pid in pred.prot_id.unique()}, name="plddt")
    tertile = pd.qcut(plddt, 3, labels=["low", "mid", "high"])
    pred["stratum"] = pred.prot_id.map(tertile)
    assert pred.stratum.notna().all() and plddt.size == pred.prot_id.nunique()

    rows, gap_reps, gap_point = [], {}, {}
    for k, (s, d) in enumerate(pred.groupby("stratum", observed=True)):
        d = d.reset_index(drop=True)
        prot = pd.factorize(d.prot_id)[0]
        y = d.y_small_binding.to_numpy()
        point, reps = auprc_reps(y, prot, {m: d[f"p_small__{m}"].to_numpy() for m in METHODS},
                                 args.boot, args.seed + k)
        ids = d.prot_id.unique()
        info = {"stratum": s, "n_proteins": len(ids), "n_TestSet300": int(d.drop_duplicates("prot_id").split.eq(
            "TestSet300").sum()), "n_residues": len(y), "n_positive": int(np.sum(y)), "prevalence": float(np.mean(y)),
            "plddt_min": float(plddt[ids].min()), "plddt_max": float(plddt[ids].max()),
            "plddt_mean": float(plddt[ids].mean())}
        for m in METHODS:
            rows.append({**info, "kind": "AUPRC", "method": m, "value": point[m], **spread(reps[m])})
        for a, b in GAPS:
            g = reps[a] - reps[b]
            gap_reps[(s, a)], gap_point[(s, a)] = g, point[a] - point[b]
            rows.append({**info, "kind": "gap", "method": f"{a} - {b}", "value": point[a] - point[b], **spread(g)})

    inter = []
    for a, b in GAPS:
        g = gap_reps[("low", a)] - gap_reps[("high", a)]
        inter.append({"gap": f"{a} - {b}", "low_tertile": gap_point[("low", a)], "high_tertile": gap_point[("high", a)],
                      "interaction_low_minus_high": gap_point[("low", a)] - gap_point[("high", a)], **spread(g)})

    dest = ROOT / "reports" / "p2rank"
    dest.mkdir(parents=True, exist_ok=True)
    strata = pd.DataFrame(rows)
    strata.to_csv(dest / "p2rank_small_plddt_strata.csv", index=False)
    pd.DataFrame(inter).to_csv(dest / "p2rank_small_plddt_interaction.csv", index=False)

    pd.set_option("display.width", 220)
    cols = ["stratum", "kind", "method", "value", "sd", "ci_lo", "ci_hi", "n_proteins", "n_TestSet300", "n_positive",
            "prevalence", "plddt_min", "plddt_max"]
    print(strata[cols].round(3).to_string(index=False))
    print("\nInteraction (gap in the lowest pLDDT tertile minus gap in the highest)")
    print(pd.DataFrame(inter).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
