#!/usr/bin/env python
"""Do metal-ion and small-molecule binding sites co-localise, and how well does each method predict both?

Part 1 (labels and ESMFold structures, data/zenodo_adabind_data_v1): proteins with both a metal-binding and a
small-molecule-binding residue; residues that carry both labels; C-alpha distances between metal-binding residues and the
nearest small-molecule-binding residue, with and without residues that carry both labels.
Part 2 (reports/finalize_results/predictions/test_residue_predictions.csv.gz): on the proteins with both labels, residue-level
AUPRC of every method for the metal and the small-molecule task, mean +/- SD (95% CI) over protein bootstrap resamples, and the
paired difference AdaBind minus each method.

    python scripts/metal_small_colocalisation.py [--boot 2000] [--seed 2026]
Outputs: reports/metal_small_colocalisation/{colocalisation_per_protein,colocalisation_summary,dual_site_auprc}.csv
"""
import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "zenodo_adabind_data_v1"
LABELS = {"TestSet300": DATA / "labels/testset300_binding_sites.csv",
          "TestSetNew46": DATA / "labels/testsetnew46_binding_sites.csv"}
METHODS = {"metal": ["AdaBind", "ESMBind", "bindNode24", "bindEmbed21DL", "MetalBind"],
           "small": ["AdaBind", "bindNode24", "bindEmbed21DL"]}


def ca_coords(fp):
    model = next(PDBParser(QUIET=True).get_structure("p", fp).get_models())
    chain = model["A"] if "A" in model else next(model.get_chains())
    return np.array([r["CA"].coord for r in chain if is_aa(r, standard=True) and "CA" in r])


def colocalisation(split, fp):
    df = pd.read_csv(fp)
    rows = []
    for r in df.itertuples():
        m = np.flatnonzero(np.array(ast.literal_eval(r.metal_binding)))
        s = np.flatnonzero(np.array(ast.literal_eval(r.small_binding)))
        if not (len(m) and len(s)):
            continue
        shared = np.intersect1d(m, s)
        row = {"split": split, "prot_id": r.prot_id, "length": r.sequence_length, "n_metal": len(m), "n_small": len(s),
               "n_shared": len(shared)}
        pdb = DATA / "structures/esmfold" / f"{r.prot_id}.pdb"
        if pdb.exists():
            ca = ca_coords(pdb)
            if len(ca) == r.sequence_length:
                d = np.linalg.norm(ca[m][:, None] - ca[s][None], axis=-1)
                row.update({"closest_pair": d.min(), "metal_to_nearest_small": d.min(axis=1).mean()})
                m_only, s_only = np.setdiff1d(m, shared), np.setdiff1d(s, shared)
                if len(m_only) and len(s_only):
                    d2 = np.linalg.norm(ca[m_only][:, None] - ca[s_only][None], axis=-1)
                    row.update({"closest_pair_unshared": d2.min(), "metal_to_nearest_small_unshared": d2.min(axis=1).mean()})
        rows.append(row)
    return pd.DataFrame(rows), len(df)


def dual_site(pred, boot, seed):
    out = []
    for split, d in pred.groupby("split"):
        both = d.groupby("prot_id").agg(m=("y_metal_binding", "sum"), s=("y_small_binding", "sum"))
        keep = both[(both.m > 0) & (both.s > 0)].index
        d = d[d.prot_id.isin(keep)].reset_index(drop=True)
        prot = pd.factorize(d.prot_id)[0]
        n = prot.max() + 1
        pos = [np.flatnonzero(prot == p) for p in range(n)]
        rng = np.random.default_rng(seed)
        draws = [np.concatenate([pos[p] for p in rng.integers(0, n, n)]) for _ in range(boot)]
        for task, methods in METHODS.items():
            y = d[f"y_{task}_binding"].to_numpy()
            score = {m: d[f"p_{task}__{m}"].to_numpy() for m in methods}
            ok = [i for i in draws if y[i].sum() > 0]
            reps = {m: np.array([average_precision_score(y[i], score[m][i]) for i in ok]) for m in methods}
            point = {m: average_precision_score(y, score[m]) for m in methods}
            for m in methods:
                r = reps[m]
                row = {"split": split, "task": task, "method": m, "n_proteins": n, "n_positive": int(y.sum()),
                       "AUPRC": point[m], "sd": r.std(ddof=1), "ci_lo": np.percentile(r, 2.5), "ci_hi": np.percentile(r, 97.5)}
                if m != "AdaBind":
                    diff = reps["AdaBind"] - r
                    row.update({"diff_AdaBind_minus": point["AdaBind"] - point[m], "diff_sd": diff.std(ddof=1),
                                "diff_ci_lo": np.percentile(diff, 2.5), "diff_ci_hi": np.percentile(diff, 97.5)})
                out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(ROOT / "reports" / "metal_small_colocalisation"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    per, summ = [], []
    for split, fp in LABELS.items():
        t, n = colocalisation(split, fp)
        per.append(t)

        def sd(v):
            v = v.dropna()
            return v.std(ddof=1) if len(v) > 1 else np.nan
        summ.append({
            "split": split, "n_proteins": n, "n_both_labels": len(t), "n_with_shared_residue": int((t.n_shared > 0).sum()),
            "n_with_structure": int(t.closest_pair.notna().sum()), "closest_pair_le_8A": int((t.closest_pair <= 8).sum()),
            "metal_to_nearest_small_mean": t.metal_to_nearest_small.mean(), "metal_to_nearest_small_sd": sd(t.metal_to_nearest_small),
            "metal_to_nearest_small_median": t.metal_to_nearest_small.median(),
            "closest_pair_mean": t.closest_pair.mean(), "closest_pair_sd": sd(t.closest_pair),
            "closest_pair_median": t.closest_pair.median(),
            "n_unshared_analysed": int(t.metal_to_nearest_small_unshared.notna().sum()),
            "metal_to_nearest_small_unshared_mean": t.metal_to_nearest_small_unshared.mean(),
            "metal_to_nearest_small_unshared_sd": sd(t.metal_to_nearest_small_unshared),
            "metal_to_nearest_small_unshared_median": t.metal_to_nearest_small_unshared.median(),
            "shared_fraction_of_metal_residues": t.n_shared.sum() / t.n_metal.sum(),
            "shared_fraction_of_small_residues": t.n_shared.sum() / t.n_small.sum()})
    per = pd.concat(per, ignore_index=True)
    per.to_csv(out / "colocalisation_per_protein.csv", index=False)
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "colocalisation_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.round(3).T.to_string())

    pred = pd.read_csv(ROOT / "reports/finalize_results/predictions/test_residue_predictions.csv.gz")
    pred["prot_id"] = pred.prot_id.astype(str)
    ds = dual_site(pred, args.boot, args.seed)
    ds.to_csv(out / "dual_site_auprc.csv", index=False)
    print(ds.round(3).to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
