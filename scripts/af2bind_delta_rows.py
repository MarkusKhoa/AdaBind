#!/usr/bin/env python
"""Rows for Supplementary Table S3 (paired AUPRC difference, AdaBind minus the method in each row), small molecules.

Adds the comparators that tables/delta_auprc.tex (notebook 16's bootstrap of the published predictors) does not contain:
  P2Rank                                  all residues, P2Rank 2.5.1 on the ESMFold structures
  AF2BIND                                 all residues
  AF2BIND, ligand-size-matched labels     negatives + positives that contact a ligand of 10-200 heavy atoms (scripts/
                                          af2bind_label_matched.py); other positives are left out
  AF2BIND, training-dissimilar proteins   test proteins with < 30% sequence identity to every AF2BIND training chain
                                          (mmseqs2, data/external/af2bind/train_overlap.csv)
Each for TestSet300, TestSetNew46 and both sets pooled; protein bootstrap B = 10,000, seed 2026 (the resampling of
scripts/finalize_metrics.py), 95% percentile interval.

    conda run -n AdaBind python scripts/af2bind_delta_rows.py

Output: reports/p2rank/af2bind_delta_rows.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from af2bind_compare import P2_TEST, PRED, af2bind_column, paired_curve  # noqa: E402
from p2rank_baseline import load_p2rank  # noqa: E402

OVERLAP = ROOT / "data/external/af2bind/train_overlap.csv"
CATEGORIES = ROOT / "reports/p2rank/af2bind_label_matched_residues.csv"


def main():
    pred = pd.read_csv(PRED)
    pred["prot_id"] = pred.prot_id.astype(str)
    pred["p_af"] = af2bind_column(pred)
    pred["p_p2"] = load_p2rank(P2_TEST, pred)
    ov = pd.read_csv(OVERLAP, dtype={"prot_id": str})[["prot_id", "max_id"]]
    cat = pd.read_csv(CATEGORIES, dtype={"prot_id": str})[["prot_id", "resid", "category"]]
    pred = pred.merge(ov, on="prot_id", how="left").merge(cat, on=["prot_id", "resid"], how="left")
    assert pred.max_id.notna().all() and pred.category.notna().all()

    variants = {
        "P2Rank": ("p_p2", pred.index == pred.index),
        "AF2BIND": ("p_af", pred.index == pred.index),
        "AF2BIND, ligand-size-matched labels": ("p_af", pred.category.isin(["negative", "large ligand"])),
        "AF2BIND, training-dissimilar proteins": ("p_af", pred.max_id < 0.3),
    }
    rows = []
    for name, (col, keep) in variants.items():
        for scope in ("TestSet300", "TestSetNew46", "Pooled test"):
            sel = keep & (pred.split == scope if scope != "Pooled test" else True)
            d = pred[sel].reset_index(drop=True)
            y = d.y_small_binding.to_numpy()
            prot = pd.factorize(d.prot_id)[0]
            r = paired_curve(y, prot, d.p_small__AdaBind.to_numpy(), d[col].to_numpy(), 10000, 2026)[0]
            rows.append({"comparator": name, "scope": scope, "n_protein": int(prot.max()) + 1, "n_positive": int(y.sum()),
                         "diff_AdaBind_minus": r["diff"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"]})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "reports/p2rank/af2bind_delta_rows.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
