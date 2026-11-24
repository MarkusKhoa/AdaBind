#!/usr/bin/env python
"""Run AF2BIND (Gazizov et al., bioRxiv 2023, https://doi.org/10.1101/2023.10.15.562410) as released on the ESMFold
structures of TestSet300 and TestSetNew46. Small-molecule branch only.

Run inside the conda env `AF2BIND` (JAX + ColabDesign v1.1.1), not `AdaBind`:

    conda run -n AF2BIND python scripts/af2bind_infer.py [--pids A,B] [--limit 5] [--seed 0]

The af2bind() function and the input preparation are copied from the authors' notebook (af2bind.ipynb, MIT licence)
so that the predictions are those of the released model: target backbone from the PDB file with side chains masked
(rm_target_sc=True), target sequence kept, 20 one-residue "bait" chains, one pass without MSA, logistic regression
on the pair representation. Default model: AF2-pair, no side chains, seed 0 (the notebook default).

Output: data/external/af2bind/out/<prot_id>.csv (resid, p_bind) and out/run_log.csv (time, length, status per protein).
Existing outputs are skipped, so the run can be resumed.
"""
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AF2B = ROOT / "data/external/af2bind"
PDB_DIR = ROOT / "data/zenodo_adabind_data_v1/structures/esmfold"
PRED = ROOT / "reports/finalize_results/predictions/test_residue_predictions.csv.gz"


class ParamsUnpickler(pickle.Unpickler):
    # params were pickled as jax arrays, which newer jax can't restore; load them as numpy (as in the notebook)
    def find_class(self, module, name):
        if (module, name) == ("jax._src.array", "_reconstruct_array"):
            def reconstruct_array(fun, args, arr_state, aval_state):
                x = fun(*args)
                x.__setstate__(arr_state)
                return x
            return reconstruct_array
        return super().find_class(module, name)


def load_head(seed: int, mask_sidechains: bool = True):
    import jax
    name = f"split_nosc_pair_A_split_nosc_pair_B_{seed}" if mask_sidechains else f"split_pair_A_split_pair_B_{seed}"
    with open(AF2B / "repo/af2bind_params/attempt_7_2k_lam0-03" / f"{name}.pickle", "rb") as fh:
        p = ParamsUnpickler(fh).load()
    p = dict(**p["~"], **p["linear"])
    return jax.tree_util.tree_map(lambda x: np.asarray(x), p)


def af2bind(outputs, head):
    from scipy.special import expit as sigmoid
    pair_A = outputs["representations"]["pair"][:-20, -20:]
    pair_B = outputs["representations"]["pair"][-20:, :-20].swapaxes(0, 1)
    pair_A = pair_A.reshape(pair_A.shape[0], -1)
    pair_B = pair_B.reshape(pair_B.shape[0], -1)
    x = np.concatenate([pair_A, pair_B], -1)
    x = (x - head["mean"]) / head["std"]
    x = (x * head["w"][:, 0]) + (head["b"] / x.shape[-1])
    p_bind_aa = x.reshape(x.shape[0], 2, 20, -1).sum((1, 3))
    return sigmoid(p_bind_aa.sum(-1))


def predict(pdb_filename, head, mask_sidechains=True):
    from colabdesign import mk_afdesign_model, clear_mem
    clear_mem()
    m = mk_afdesign_model(protocol="binder", debug=True, data_dir=str(AF2B))
    m.prep_inputs(pdb_filename=str(pdb_filename), chain="A", binder_len=20, rm_target_sc=mask_sidechains,
                  rm_target_seq=False)
    r_idx = m._inputs["residue_index"][-20] + (1 + np.arange(20)) * 50
    m._inputs["residue_index"][-20:] = r_idx.flatten()
    m.set_seq("ACDEFGHIKLMNPQRSTVWY")
    m.predict(verbose=False)
    p = af2bind(m.aux["debug"]["outputs"], head)
    resid = np.asarray(m._pdb["idx"]["residue"]).astype(int)
    return resid, p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pids", default="", help="comma-separated protein ids (default: all of TestSet300 + TestSetNew46)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N proteins (timing test)")
    ap.add_argument("--seed", type=int, default=0, help="which of the 10 released heads (notebook default 0)")
    ap.add_argument("--max-len", type=int, default=0, help="skip proteins longer than this (0 = no limit)")
    ap.add_argument("--out", default=str(AF2B / "out"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(PRED, usecols=["split", "prot_id", "resid"])
    pred["prot_id"] = pred.prot_id.astype(str)
    lens = pred.groupby("prot_id", sort=False).resid.agg(["size", "min", "max"])
    pids = [p for p in args.pids.split(",") if p] or list(lens.index)
    if args.limit:
        pids = pids[:args.limit]
    head = load_head(args.seed)

    log = out / "run_log.csv"
    for pid in pids:
        dest = out / f"{pid}.csv"
        if dest.exists():
            continue
        L = int(lens.loc[pid, "size"])
        if args.max_len and L > args.max_len:
            continue
        t0 = time.time()
        try:
            resid, p = predict(PDB_DIR / f"{pid}.pdb", head)
            assert len(resid) == L, (pid, len(resid), L)
            assert set(resid) == set(range(int(lens.loc[pid, "min"]), int(lens.loc[pid, "max"]) + 1)), pid
            pd.DataFrame({"resid": resid, "p_bind": p}).to_csv(dest, index=False)
            status = "ok"
        except Exception as e:  # keep going; failures are reported, not hidden
            status = f"FAIL {type(e).__name__}: {str(e)[:120]}"
        rec = pd.DataFrame([{"prot_id": pid, "length": L, "seconds": round(time.time() - t0, 1), "status": status}])
        rec.to_csv(log, mode="a", header=not log.exists(), index=False)
        print(rec.iloc[0].to_dict(), flush=True)


if __name__ == "__main__":
    main()
