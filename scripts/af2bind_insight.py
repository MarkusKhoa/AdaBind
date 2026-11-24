#!/usr/bin/env python
"""Where does AdaBind differ from AF2BIND and the other small-molecule predictors? (analyses A1-A3 and A7)

All tests were declared before they were run; a finding is written into the manuscript only if its 95% interval excludes
zero (for A2-A3 after Holm correction over the four tests listed below).

A1  Top-N recovery (the metric of the AF2BIND paper): for every protein with at least one small-molecule-binding residue,
    the fraction of its N true positives among its N highest-scoring residues (N = number of positives). Mean over
    proteins; paired protein bootstrap for the difference to AdaBind.
A2  Per-protein win/loss of AdaBind against AF2BIND on that recovery; gap (AdaBind - AF2BIND) in tertiles of protein
    length and of mean ESMFold pLDDT, and the interaction (lowest minus highest tertile).            [Holm tests 1, 2]
A3  At the fixed threshold 0.4901: number of residues predicted positive divided by the number of true positives, and the
    share of leucine, valine and isoleucine among the false positives (hypothesis: AF2BIND over-predicts aliphatic
    surface residues), AF2BIND minus AdaBind.                                                        [Holm tests 3, 4]
A7  TestSetNew46 without the protein that carries about a third of its small-molecule-binding residues: pooled AUPRC and
    AUROC of AF2BIND, AdaBind and P2Rank and the paired differences.

Protein-level bootstrap, B = 10,000, seed 2026, same resamples for every statistic of a split.

    conda run -n AdaBind python scripts/af2bind_insight.py

Outputs: reports/p2rank/af2bind_insight_{a1_topn,a2_strata,a3_errors,a7_ts46_without_top_protein}.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import finalize_metrics as FM  # noqa: E402
from af2bind_compare import P2_TEST, PRED, af2bind_column, paired_curve  # noqa: E402
from p2rank_baseline import load_p2rank  # noqa: E402
from p2rank_plddt_strata import mean_plddt  # noqa: E402

THRESHOLD = 0.4901010101010101  # AdaBind small-molecule threshold (manifest.json); also used for AF2BIND
METHODS = {"AdaBind": "p_small__AdaBind", "bindNode24": "p_small__bindNode24", "bindEmbed21DL": "p_small__bindEmbed21DL",
           "P2Rank": "p_p2", "AF2BIND": "p_af"}
ALIPHATIC = set("LVI")


def resamples(n, boot, seed):
    """B x n matrix of how often each protein enters a replicate (same rule as finalize_metrics.score)."""
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(boot)]).astype(float)


def summarise(point, reps):
    r = np.asarray(reps, float)
    p = 2 * min(np.mean(r <= 0), np.mean(r >= 0))
    return {"value": point, "sd": float(np.std(r, ddof=1)), "ci_lo": float(np.percentile(r, 2.5)),
            "ci_hi": float(np.percentile(r, 97.5)), "p_boot": float(min(1.0, max(p, 1.0 / len(r))))}


def wmean(C, num, den):
    """Weighted ratio sum(c*num)/sum(c*den) for every replicate (rows of C) and for the original sample."""
    return (C @ num) / (C @ den), num.sum() / den.sum()


def protein_table(pred):
    rows = []
    for pid, g in pred.groupby("prot_id", sort=False):
        y = g.y_small_binding.to_numpy()
        n = int(y.sum())
        r = {"prot_id": pid, "split": g.split.iloc[0], "L": len(g), "n_pos": n, "has_pos": float(n > 0)}
        aa = g.aa.to_numpy()
        ali = np.array([a in ALIPHATIC for a in aa])
        r["n_neg"] = int((y == 0).sum())
        r["n_neg_ali"] = int(((y == 0) & ali).sum())
        for m, c in METHODS.items():
            p = g[c].to_numpy()
            if n > 0:
                top = np.argsort(-p, kind="mergesort")[:n]
                r[f"rec_{m}"] = float(y[top].sum() / n)
            else:
                r[f"rec_{m}"] = 0.0
            pos = p >= THRESHOLD if m != "P2Rank" else p >= 0.4586
            if n > 0:  # false positives among the N top-ranked residues: no threshold involved
                r[f"topfp_{m}"] = int((y[top] == 0).sum())
                r[f"topfp_ali_{m}"] = int(((y[top] == 0) & ali[top]).sum())
            else:
                r[f"topfp_{m}"] = r[f"topfp_ali_{m}"] = 0
            r[f"npred_{m}"] = int(pos.sum())
            r[f"fp_{m}"] = int((pos & (y == 0)).sum())
            r[f"fp_ali_{m}"] = int((pos & (y == 0) & ali).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    dest = ROOT / "reports" / "p2rank"

    pred = pd.read_csv(PRED)
    pred["prot_id"] = pred.prot_id.astype(str)
    pred["p_af"] = af2bind_column(pred)
    pred["p_p2"] = load_p2rank(P2_TEST, pred)
    d = protein_table(pred)
    d["plddt"] = d.prot_id.map({p: mean_plddt(p) for p in d.prot_id})
    pos = d[d.has_pos == 1]
    d["len_tert"] = pd.qcut(pos.L, 3, labels=["short", "middle", "long"]).reindex(d.index)
    d["plddt_tert"] = pd.qcut(pos.plddt, 3, labels=["low", "middle", "high"]).reindex(d.index)

    # ---------------------------------------------------------------- A1 top-N recovery, pooled and per split
    a1 = []
    scopes = {"both test sets": d.split.notna(), "TestSet300": d.split == "TestSet300", "TestSetNew46": d.split == "TestSetNew46"}
    boots = {}
    for name, sel in scopes.items():
        dd = d[sel].reset_index(drop=True)
        C = resamples(len(dd), args.boot, args.seed)
        boots[name] = (dd, C)
        hp = dd.has_pos.to_numpy()
        for m in METHODS:
            rb, r0 = wmean(C, hp * dd[f"rec_{m}"].to_numpy(), hp)
            a1.append({"scope": name, "method": m, "n_protein_with_positives": int(hp.sum()), "metric": "top-N recovery",
                       **{k: v for k, v in summarise(r0, rb).items() if k != "p_boot"}})
            if m != "AF2BIND":
                ab, a0 = wmean(C, hp * dd["rec_AF2BIND"].to_numpy(), hp)
                a1.append({"scope": name, "method": f"AF2BIND minus {m}", "n_protein_with_positives":
                           int(hp.sum()), "metric": "top-N recovery difference", **{k: v for k, v in
                           summarise(a0 - r0, ab - rb).items() if k != "p_boot"}})
    pd.DataFrame(a1).to_csv(dest / "af2bind_insight_a1_topn.csv", index=False)

    # ---------------------------------------------------------------- A2 win/loss and strata (both test sets)
    dd, C = boots["both test sets"]
    hp = dd.has_pos.to_numpy()
    gap = dd.rec_AdaBind.to_numpy() - dd.rec_AF2BIND.to_numpy()
    a2, tests = [], {}
    for lab, v in (("AdaBind better than AF2BIND (share of proteins)", (gap > 0).astype(float)),
                   ("AF2BIND better than AdaBind (share of proteins)", (gap < 0).astype(float)),
                   ("tie (share of proteins)", (gap == 0).astype(float))):
        b, p0 = wmean(C, hp * v, hp)
        a2.append({"quantity": lab, "stratum": "all", **summarise(p0, b)})
    for strat, col, lo, hi in (("length", "len_tert", "short", "long"), ("pLDDT", "plddt_tert", "low", "high")):
        gaps = {}
        for t in dd[col].dropna().unique():
            m = (dd[col] == t).to_numpy().astype(float)
            b, p0 = wmean(C, m * gap, m)
            gaps[t] = (b, p0)
            a2.append({"quantity": "gap in recovery (AdaBind - AF2BIND)", "stratum": f"{strat}: {t}",
                       "n_protein": int(m.sum()), **summarise(p0, b)})
        ib, i0 = gaps[lo][0] - gaps[hi][0], gaps[lo][1] - gaps[hi][1]
        row = {"quantity": f"interaction ({lo} minus {hi})", "stratum": strat, **summarise(i0, ib)}
        a2.append(row)
        tests[f"{strat} interaction"] = row["p_boot"]

    # ---------------------------------------------------------------- A3 over-prediction and false-positive profile
    npos = dd.n_pos.to_numpy().astype(float)
    rb = {m: wmean(C, dd[f"npred_{m}"].to_numpy().astype(float), npos) for m in ("AdaBind", "AF2BIND", "P2Rank")}
    a3 = []
    for m, (b, p0) in rb.items():
        a3.append({"quantity": "predicted positives / true positives", "method": m, **{k: v for k, v in
                   summarise(p0, b).items() if k != "p_boot"}})
    s = summarise(rb["AF2BIND"][1] - rb["AdaBind"][1], rb["AF2BIND"][0] - rb["AdaBind"][0])
    a3.append({"quantity": "predicted/true, AF2BIND minus AdaBind", "method": "difference", **s})
    tests["predicted/true ratio"] = s["p_boot"]
    sb = {m: wmean(C, dd[f"fp_ali_{m}"].to_numpy().astype(float), dd[f"fp_{m}"].to_numpy().astype(float))
          for m in ("AdaBind", "AF2BIND")}
    bg = wmean(C, dd.n_neg_ali.to_numpy().astype(float), dd.n_neg.to_numpy().astype(float))
    for m, (b, p0) in sb.items():
        a3.append({"quantity": "share of L, V, I among false positives", "method": m, **{k: v for k, v in
                   summarise(p0, b).items() if k != "p_boot"}})
    a3.append({"quantity": "share of L, V, I among all negative residues (background)", "method": "all residues",
               **{k: v for k, v in summarise(bg[1], bg[0]).items() if k != "p_boot"}})
    s = summarise(sb["AF2BIND"][1] - sb["AdaBind"][1], sb["AF2BIND"][0] - sb["AdaBind"][0])
    a3.append({"quantity": "L, V, I share among false positives, AF2BIND minus AdaBind", "method": "difference", **s})
    tests["aliphatic false-positive share"] = s["p_boot"]
    # sensitivity without a threshold: false positives among the N top-ranked residues of each protein
    tb = {m: wmean(C, dd[f"topfp_ali_{m}"].to_numpy().astype(float), dd[f"topfp_{m}"].to_numpy().astype(float))
          for m in ("AdaBind", "AF2BIND")}
    for m, (b, p0) in tb.items():
        a3.append({"quantity": "share of L, V, I among false positives in the top N (no threshold)", "method": m,
                   **{k: v for k, v in summarise(p0, b).items() if k != "p_boot"}})
    a3.append({"quantity": "same, AF2BIND minus AdaBind (sensitivity, not in the Holm family)", "method": "difference",
               **summarise(tb["AF2BIND"][1] - tb["AdaBind"][1], tb["AF2BIND"][0] - tb["AdaBind"][0])})

    # Holm correction over the four declared tests
    order = sorted(tests, key=tests.get)
    holm, run = {}, 0.0
    for k, name in enumerate(order):
        run = max(run, min(1.0, (len(order) - k) * tests[name]))
        holm[name] = run
    a2 = pd.DataFrame(a2)
    a3 = pd.DataFrame(a3)
    a2["p_holm"] = np.nan
    a3["p_holm"] = np.nan
    inter = a2.quantity.str.startswith("interaction")
    a2.loc[inter, "p_holm"] = [holm[f"{s} interaction"] for s in a2[inter].stratum]
    a3.loc[a3.quantity == "predicted/true, AF2BIND minus AdaBind", "p_holm"] = holm["predicted/true ratio"]
    a3.loc[a3.quantity.str.startswith("L, V, I share among false positives, AF2BIND"), "p_holm"] = holm[
        "aliphatic false-positive share"]
    a2.to_csv(dest / "af2bind_insight_a2_strata.csv", index=False)
    a3.to_csv(dest / "af2bind_insight_a3_errors.csv", index=False)

    # ---------------------------------------------------------------- A7 TestSetNew46 without its dominant protein
    # the three PSI-LHCI chains of Pisum sativum (UniProt: Q32904 Lhca3, E1C9L3 PsaK, P17227 PsaI) that the case-study
    # section reports as 143 of the 425 small-molecule-binding residues (34%)
    complex_ids = ["Q32904", "E1C9L3", "P17227"]
    t46 = d[d.split == "TestSetNew46"]
    n_c = int(t46[t46.prot_id.isin(complex_ids)].n_pos.sum())
    assert n_c == 143, n_c
    top = pd.Series({"prot_id": "+".join(complex_ids)})
    print(f"\nA7: PSI-LHCI chains {complex_ids}: {n_c} of {int(t46.n_pos.sum())} positives ({n_c / t46.n_pos.sum():.1%})")
    sub = pred[(pred.split == "TestSetNew46") & (~pred.prot_id.isin(complex_ids))].reset_index(drop=True)
    y = sub.y_small_binding.to_numpy()
    prot = pd.factorize(sub.prot_id)[0]
    models = {m: (sub[c].to_numpy(), THRESHOLD) for m, c in METHODS.items() if m in ("AF2BIND", "AdaBind", "P2Rank")}
    a7 = [{"quantity": r["metric"], "method": r["method"], **{k: r[k] for k in ("value", "sd", "ci_lo", "ci_hi")},
           "n_protein": int(prot.max()) + 1, "n_positive": int(y.sum())}
          for r in FM.score(y, prot, models, args.boot, args.seed) if r["metric"] in ("AUPRC", "AUROC")]
    for comp in ("AdaBind", "P2Rank"):
        for r in paired_curve(y, prot, models["AF2BIND"][0], models[comp][0], args.boot, args.seed):
            a7.append({"quantity": f"AF2BIND - {comp}: {r['metric']}", "method": "paired difference",
                       **{k: r[k] for k in ("sd", "ci_lo", "ci_hi")}, "value": r["diff"]})
    a7 = pd.DataFrame(a7)
    a7.insert(0, "excluded_protein", top.prot_id)
    a7.to_csv(dest / "af2bind_insight_a7_ts46_without_top_protein.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 70)
    for name, df in (("A1 top-N recovery", pd.DataFrame(a1)), ("A2 win/loss and strata", a2), ("A3 errors", a3),
                     ("A7 TestSetNew46 without top protein", a7)):
        print(f"\n== {name}")
        print(df.round(3).to_string(index=False))
    print("\nHolm-adjusted p of the four declared tests:", {k: round(v, 4) for k, v in holm.items()})


if __name__ == "__main__":
    main()
