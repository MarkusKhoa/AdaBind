#!/usr/bin/env python
"""Score P2Rank as a residue-level baseline against AdaBind on TestSet300 and TestSetNew46.

Tasks (``--task``): ``small`` (default; the comparison for the small-molecule branch) and ``metal`` (extra
measurement: how much P2Rank's residue scores carry for metal-binding labels; the metal comparators are ESMBind and
MetalBind).

Run P2Rank first (Java 17+; P2Rank 2.5.1 unpacked in data/external/p2rank/p2rank_2.5.1), on the ESMFold structures listed
in data/external/p2rank/test_esmfold.ds, with the configuration for predicted structures. P2Rank splits every line of
the .ds file on whitespace, so the paths in it must not contain spaces (symlink the PDB files into a space-free
directory if the repository lives in one):

    cd data/external/p2rank/p2rank_2.5.1
    ./prank predict -c alphafold ../test_esmfold.ds -o ../out_alphafold -threads 8

then:

    python scripts/p2rank_baseline.py [--task small|metal] [--boot 10000] [--seed 2026]

P2Rank writes <protein>.pdb_residues.csv with a per-residue ligandability `probability`; residues are matched to the
benchmark by their position in chain A (ESMFold numbering starts at 1). P2Rank is a pocket detector trained on its own
data sets, which were not filtered for homology against these test proteins.
Output: reports/p2rank/p2rank_<task>_auprc.csv (AUPRC, AUROC, prevalence, bootstrap SD and 95% CI, paired AUPRC
difference AdaBind minus P2Rank; protein-level bootstrap, same resamples for every method).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

# label column in the prediction table and the methods scored for each task (P2Rank is always added last)
TASKS = {
    "small": {"label": "y_small_binding", "methods": ("AdaBind", "bindNode24", "bindEmbed21DL", "P2Rank")},
    "metal": {"label": "y_metal_binding",
              "methods": ("AdaBind", "ESMBind", "MetalBind", "bindNode24", "bindEmbed21DL", "P2Rank")},
}


def load_p2rank(out_dir, pred):
    rows = []
    for pid, g in pred.groupby("prot_id", sort=False):
        fp = out_dir / f"{pid}.pdb_residues.csv"
        if not fp.exists():
            raise FileNotFoundError(fp)
        r = pd.read_csv(fp, skipinitialspace=True)
        r.columns = [c.strip() for c in r.columns]
        r = r[r.chain.astype(str).str.strip() == "A"] if "A" in set(r.chain.astype(str).str.strip()) else r
        prob = dict(zip(r.residue_label.astype(int), r.probability.astype(float)))
        rows.append(pd.Series([prob.get(int(i), 0.0) for i in g.resid], index=g.index))
    return pd.concat(rows).sort_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p2rank-out", default=str(ROOT / "data/external/p2rank/out_alphafold"))
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--task", choices=sorted(TASKS), default="small")
    args = ap.parse_args()
    task = TASKS[args.task]
    pred = pd.read_csv(ROOT / "reports/finalize_results/predictions/test_residue_predictions.csv.gz")
    pred["prot_id"] = pred.prot_id.astype(str)
    pred[f"p_{args.task}__P2Rank"] = load_p2rank(Path(args.p2rank_out), pred)
    out, pairwise = [], []
    for split, d in pred.groupby("split"):
        d = d.reset_index(drop=True)
        y = d[task["label"]].to_numpy()
        prot = pd.factorize(d.prot_id)[0]
        n = prot.max() + 1
        pos = [np.flatnonzero(prot == p) for p in range(n)]
        rng = np.random.default_rng(args.seed)
        draws = [i for i in (np.concatenate([pos[p] for p in rng.integers(0, n, n)]) for _ in range(args.boot))
                 if y[i].sum() > 0]
        reps = {}
        rows = {}
        for m in task["methods"]:
            p = d[f"p_{args.task}__{m}"].to_numpy()
            reps[m] = np.array([average_precision_score(y[i], p[i]) for i in draws])
            rows[m] = {"split": split, "method": m, "AUPRC": average_precision_score(y, p),
                       "AUROC": roc_auc_score(y, p), "prevalence": float(y.mean()),
                       "sd": reps[m].std(ddof=1), "ci_lo": np.percentile(reps[m], 2.5),
                       "ci_hi": np.percentile(reps[m], 97.5)}
            out.append(rows[m])
        diff = reps["AdaBind"] - reps["P2Rank"]  # looked up by name, not by row position
        rows["P2Rank"].update({"diff_AdaBind_minus": rows["AdaBind"]["AUPRC"] - rows["P2Rank"]["AUPRC"],
                               "diff_sd": diff.std(ddof=1), "diff_ci_lo": np.percentile(diff, 2.5),
                               "diff_ci_hi": np.percentile(diff, 97.5)})
        for m in task["methods"]:  # every method minus P2Rank, on the same resamples
            if m != "P2Rank":
                dm = reps[m] - reps["P2Rank"]
                pairwise.append({"split": split, "method": m, "minus": "P2Rank",
                                 "diff_AUPRC": rows[m]["AUPRC"] - rows["P2Rank"]["AUPRC"], "sd": dm.std(ddof=1),
                                 "ci_lo": np.percentile(dm, 2.5), "ci_hi": np.percentile(dm, 97.5)})
    res = pd.DataFrame(out)
    dest = ROOT / "reports" / "p2rank"
    dest.mkdir(parents=True, exist_ok=True)
    res.to_csv(dest / f"p2rank_{args.task}_auprc.csv", index=False)
    pd.DataFrame(pairwise).to_csv(dest / f"p2rank_{args.task}_pairwise.csv", index=False)
    print(pd.DataFrame(pairwise).round(3).to_string(index=False))
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
