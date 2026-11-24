#!/usr/bin/env python
"""Reduced A5: is the AF2BIND result sensitive to which released head, or which AlphaFold parameter version, is used?

AF2BIND = frozen AlphaFold2 (pair representation) + a logistic head. The authors release 10 heads (seeds 0-9, one per
cross-validation split of their own data); the main comparison uses head 0 (notebook default) and AlphaFold parameters
2022-03-02 (the version their README gives for the experiments). The README and notebook also mention 2022-12-06 and
2021-07-14.

Stage 1 (env AF2BIND, GPU): on a random subset of the 346 test proteins, run AlphaFold once per protein and apply all 10
heads; on a smaller subset, repeat with the 2021-07-14 parameters (head 0).
    conda run -n AF2BIND python scripts/af2bind_sensitivity.py --stage infer [--n 100] [--n-params 60] [--seed 7]
Stage 2 (env AdaBind): pooled AUPRC and AUROC of every head and of the 2021-07-14 run on the same proteins, next to
AdaBind, with the protein bootstrap (B = 10,000, seed 2026) for the paired gap AF2BIND - AdaBind per head.
    conda run -n AdaBind python scripts/af2bind_sensitivity.py --stage analyse

Outputs: data/external/af2bind/sens/*.csv, reports/p2rank/af2bind_sensitivity.csv
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
AF2B = ROOT / "data/external/af2bind"
SENS = AF2B / "sens"
PDB_DIR = ROOT / "data/zenodo_adabind_data_v1/structures/esmfold"
PRED = ROOT / "reports/finalize_results/predictions/test_residue_predictions.csv.gz"
P2021 = AF2B / "params_2021-07-14"


def subset(n, seed):
    ids = pd.read_csv(PRED, usecols=["prot_id"]).prot_id.astype(str).unique()
    return list(np.random.default_rng(seed).choice(ids, size=n, replace=False))


def run_model(pdb, data_dir):
    """AlphaFold pair representation of target + 20 bait chains (as scripts/af2bind_infer.py); returns outputs, resid."""
    from colabdesign import clear_mem, mk_afdesign_model
    clear_mem()
    m = mk_afdesign_model(protocol="binder", debug=True, data_dir=str(data_dir))
    m.prep_inputs(pdb_filename=str(pdb), chain="A", binder_len=20, rm_target_sc=True, rm_target_seq=False)
    r_idx = m._inputs["residue_index"][-20] + (1 + np.arange(20)) * 50
    m._inputs["residue_index"][-20:] = r_idx.flatten()
    m.set_seq("ACDEFGHIKLMNPQRSTVWY")
    m.predict(verbose=False)
    return m.aux["debug"]["outputs"], np.asarray(m._pdb["idx"]["residue"]).astype(int)


def infer(args):
    import af2bind_infer as A
    SENS.mkdir(parents=True, exist_ok=True)
    heads = [A.load_head(s) for s in range(10)]
    ids = subset(args.n, args.seed)
    (SENS / "subset.txt").write_text("\n".join(ids))
    for k, pid in enumerate(ids):
        dest = SENS / f"heads_{pid}.csv"
        if dest.exists():
            continue
        t0 = time.time()
        out, resid = run_model(PDB_DIR / f"{pid}.pdb", AF2B)
        df = pd.DataFrame({"resid": resid, **{f"h{s}": A.af2bind(out, h) for s, h in enumerate(heads)}})
        df.to_csv(dest, index=False)
        print(f"heads {k + 1}/{len(ids)} {pid} {time.time() - t0:.0f}s", flush=True)
    while not (P2021 / "params/done.txt").exists():  # wait for the 2021-07-14 download
        time.sleep(10)
    for k, pid in enumerate(ids[:args.n_params]):
        dest = SENS / f"params2021_{pid}.csv"
        if dest.exists():
            continue
        out, resid = run_model(PDB_DIR / f"{pid}.pdb", P2021)
        pd.DataFrame({"resid": resid, "p_bind": A.af2bind(out, heads[0])}).to_csv(dest, index=False)
        print(f"params2021 {k + 1}/{args.n_params} {pid}", flush=True)


def analyse(args):
    import finalize_metrics as FM
    from af2bind_compare import af2bind_column, paired_curve
    ids = (SENS / "subset.txt").read_text().split()
    pred = pd.read_csv(PRED)
    pred["prot_id"] = pred.prot_id.astype(str)
    pred["p_af"] = af2bind_column(pred)
    pred = pred[pred.prot_id.isin(ids)].reset_index(drop=True)
    for s in range(10):
        col = []
        for pid, g in pred.groupby("prot_id", sort=False):
            a = pd.read_csv(SENS / f"heads_{pid}.csv").set_index("resid")[f"h{s}"]
            col.append(pd.Series(a.loc[g.resid].to_numpy(), index=g.index))
        pred[f"h{s}"] = pd.concat(col).sort_index()
    gate = float(np.max(np.abs(pred.h0 - pred.p_af)))
    print(f"head 0 of this run vs the main run: max |difference| = {gate:.2e}")
    y = pred.y_small_binding.to_numpy()
    prot = pd.factorize(pred.prot_id)[0]
    one = np.ones(len(y))
    ada = pred.p_small__AdaBind.to_numpy()
    rows = [{"model": "AdaBind", "n_protein": len(ids), "n_positive": int(y.sum()),
             "AUPRC": FM.curve_metrics(y, one, FM.prep(ada))[0], "AUROC": FM.curve_metrics(y, one, FM.prep(ada))[1]}]
    for s in range(10):
        p = pred[f"h{s}"].to_numpy()
        gap = paired_curve(y, prot, p, ada, args.boot, args.seed)
        rows.append({"model": f"AF2BIND head {s}", "n_protein": len(ids), "n_positive": int(y.sum()),
                     "AUPRC": FM.curve_metrics(y, one, FM.prep(p))[0], "AUROC": FM.curve_metrics(y, one, FM.prep(p))[1],
                     "gap_AUPRC_vs_AdaBind": gap[0]["diff"], "gap_ci_lo": gap[0]["ci_lo"], "gap_ci_hi": gap[0]["ci_hi"]})
    # AlphaFold parameter version, on the first n_params proteins of the subset
    sub = ids[:args.n_params]
    d = pred[pred.prot_id.isin(sub)].reset_index(drop=True)
    col = []
    for pid, g in d.groupby("prot_id", sort=False):
        a = pd.read_csv(SENS / f"params2021_{pid}.csv").set_index("resid").p_bind
        col.append(pd.Series(a.loc[g.resid].to_numpy(), index=g.index))
    d["p2021"] = pd.concat(col).sort_index()
    yy = d.y_small_binding.to_numpy()
    pp = pd.factorize(d.prot_id)[0]
    for name, p in (("AF2BIND head 0, params 2022-03-02 (main)", d.p_af.to_numpy()),
                    ("AF2BIND head 0, params 2021-07-14", d.p2021.to_numpy()),
                    ("AdaBind", d.p_small__AdaBind.to_numpy())):
        m = FM.curve_metrics(yy, np.ones(len(yy)), FM.prep(p))
        rows.append({"model": name, "n_protein": len(sub), "n_positive": int(yy.sum()), "AUPRC": m[0], "AUROC": m[1]})
    gap = paired_curve(yy, pp, d.p2021.to_numpy(), d.p_small__AdaBind.to_numpy(), args.boot, args.seed)[0]
    rows.append({"model": "AF2BIND 2021-07-14 minus AdaBind (AUPRC)", "n_protein": len(sub), "n_positive": int(yy.sum()),
                 "gap_AUPRC_vs_AdaBind": gap["diff"], "gap_ci_lo": gap["ci_lo"], "gap_ci_hi": gap["ci_hi"]})
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "reports/p2rank/af2bind_sensitivity.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    h = res[res.model.str.startswith("AF2BIND head")]
    print(f"AUPRC over the 10 heads: min {h.AUPRC.min():.3f}, median {h.AUPRC.median():.3f}, max {h.AUPRC.max():.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("infer", "analyse"), required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n-params", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7, help="subset seed (infer) / bootstrap seed is fixed to 2026")
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()
    if args.stage == "infer":
        infer(args)
    else:
        args.seed = 2026
        analyse(args)


if __name__ == "__main__":
    main()
