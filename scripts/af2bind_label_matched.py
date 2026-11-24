#!/usr/bin/env python
"""Does the gap between AdaBind and AF2BIND depend on how small-molecule labels are defined? (analysis A6)

AF2BIND was trained on residues that contact ligands of 10-200 heavy atoms (no ions, no crystallisation additives). The
benchmark labels of AdaBind (BioLiP) also contain residues whose only ligand is smaller. This script re-scores every
method on the residues AF2BIND is designed to find.

Per residue of the benchmark, from the BioLiP2 sites of its UniProt entry (resolution <= 2.5 A, as for the labels; the
receptor sequence is mapped onto the benchmark sequence with the helpers of biolip_shared_residue_ligands.py):
  large ligand   >= 1 contacting small-molecule ligand with 10-200 heavy atoms (heavy atoms from the RCSB chemical-component
                 formula; metal ions, DNA/RNA and peptides are not small-molecule ligands here)
  small only     positive residue contacted only by ligands outside that range
  unmapped       positive residue for which no ligand could be mapped (sequence mapping or filtering differences)
Evaluation set = all negative residues + positive residues with a large ligand. Positives of the other two kinds are
left out of the evaluation (neither positive nor negative), which can only help AF2BIND; a gap that survives is therefore
robust to this label difference. Metrics: pooled AUPRC and AUROC, protein bootstrap (B = 10,000, seed 2026), paired
differences AF2BIND minus AdaBind and minus P2Rank.

    conda run -n AdaBind python scripts/af2bind_label_matched.py [--boot 10000] [--seed 2026]

Inputs : data/external/biolip2/BioLiP.txt.gz, data/zenodo_adabind_data_v1/labels/test*.csv, AF2BIND and P2Rank outputs
Outputs: reports/p2rank/af2bind_label_matched_{residue_categories,metrics,diff}.csv
"""
import argparse
import ast
import gzip
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import biolip_shared_residue_ligands as B  # noqa: E402  (formula, mapping, METAL_IONS, paths)
import finalize_metrics as FM  # noqa: E402
from af2bind_compare import P2_TEST, PRED, af2bind_column, paired_curve  # noqa: E402
from p2rank_baseline import load_p2rank  # noqa: E402

LABELS = ROOT / "data/zenodo_adabind_data_v1/labels"
SKIP_LIGANDS = set(B.METAL_IONS) | {"DNA", "RNA", "dna", "rna", "peptide"}
MIN_HEAVY, MAX_HEAVY = 10, 200
METHODS = {"AF2BIND": "p_af", "AdaBind": "p_small__AdaBind", "bindNode24": "p_small__bindNode24",
           "bindEmbed21DL": "p_small__bindEmbed21DL", "P2Rank": "p_p2"}


def heavy_atoms(formula: str) -> int:
    n = 0
    for el, cnt in re.findall(r"([A-Z][a-z]?)(\d*)", formula or ""):
        if el not in ("H", "D"):
            n += int(cnt) if cnt else 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    lab = pd.concat([pd.read_csv(LABELS / f) for f in ("testset300_binding_sites.csv", "testsetnew46_binding_sites.csv")])
    seqs = dict(zip(lab.prot_id.astype(str), lab.sequence))
    ysmall = {str(r.prot_id): np.array(ast.literal_eval(r.small_binding)) for r in lab.itertuples()}

    # ---- BioLiP2 sites of the test proteins
    rows = []
    with gzip.open(B.BIOLIP, "rt") as fh:
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if len(c) < 21 or c[17] not in seqs:
                continue
            try:
                if float(c[2]) > 2.5:
                    continue
            except ValueError:
                continue
            rows.append({"prot_id": c[17], "pdb": c[0], "chain": c[1], "ligand": c[4], "renum": c[8], "receptor": c[20]})
    sites = pd.DataFrame(rows)
    sites = sites[~sites.ligand.isin(SKIP_LIGANDS)]
    print(f"BioLiP2 sites (<= 2.5 A, non-ion, non-nucleic-acid, non-peptide) for the test proteins: {len(sites)} "
          f"in {sites.prot_id.nunique()} proteins; {sites.ligand.nunique()} distinct ligands")

    # ---- heavy-atom count of every ligand (RCSB chemical-component formula, cached)
    cache = json.loads(B.CCD_CACHE.read_text()) if B.CCD_CACHE.exists() else {}
    todo = sorted(set(sites.ligand) - set(cache))
    if todo:
        with ThreadPoolExecutor(8) as ex:
            for lig, f in zip(todo, ex.map(lambda l: B.formula(l, {}), todo)):
                cache[lig] = f
        B.CCD_CACHE.write_text(json.dumps(cache, indent=1, sort_keys=True))
    missing = [l for l in set(sites.ligand) if not cache.get(l)]
    ha = {l: heavy_atoms(cache.get(l, "")) for l in set(sites.ligand)}
    print(f"ligands without a formula (treated as size 0, i.e. not large): {len(missing)}")

    # ---- per residue: sizes of the ligands that contact it
    cats = []
    for pid, seq in seqs.items():
        y = ysmall[pid]
        contacts = [set() for _ in seq]
        sub = sites[sites.prot_id == pid]
        for (pdb, chain, receptor), grp in sub.groupby(["pdb", "chain", "receptor"]):
            mp = B.mapping(receptor, seq)
            for s in grp.itertuples():
                for tok in s.renum.split():
                    num = re.sub(r"^[A-Za-z]+", "", tok)
                    if num.isdigit() and mp.get(int(num)) is not None:
                        contacts[mp[int(num)]].add(s.ligand)
        for i, c in enumerate(contacts):
            sizes = [ha[l] for l in c]
            large = any(MIN_HEAVY <= s <= MAX_HEAVY for s in sizes)
            kind = ("negative" if y[i] == 0 else
                    "large ligand" if large else "small only" if c else "unmapped")
            cats.append({"prot_id": pid, "resid": i + 1, "y_small": int(y[i]), "category": kind,
                         "n_ligands": len(c), "max_heavy_atoms": max(sizes) if sizes else 0})
    cat = pd.DataFrame(cats)

    pred = pd.read_csv(PRED)
    pred["prot_id"] = pred.prot_id.astype(str)
    pred["p_af"] = af2bind_column(pred)
    pred["p_p2"] = load_p2rank(P2_TEST, pred)
    pred = pred.merge(cat, on=["prot_id", "resid"], how="left")
    assert pred.category.notna().all()
    assert (pred.y_small == pred.y_small_binding).all()  # same labels as the benchmark

    summary = (pred[pred.y_small == 1].groupby(["split", "category"]).size().unstack(fill_value=0))
    summary["positives"] = summary.sum(axis=1)
    summary["fraction_large"] = summary["large ligand"] / summary["positives"]
    print(summary.to_string())

    dest = ROOT / "reports" / "p2rank"
    dest.mkdir(parents=True, exist_ok=True)
    summary.to_csv(dest / "af2bind_label_matched_residue_categories.csv")
    cat.to_csv(dest / "af2bind_label_matched_residues.csv", index=False)  # per-residue category, for diagnostics

    metric_rows, diff_rows = [], []
    for name, sel in (("TestSet300", pred.split == "TestSet300"), ("TestSetNew46", pred.split == "TestSetNew46"),
                      ("both test sets", pred.split.notna())):
        d = pred[sel & (pred.category.isin(["negative", "large ligand"]))].reset_index(drop=True)
        y = d.y_small.to_numpy()
        prot = pd.factorize(d.prot_id)[0]
        models = {m: (d[c].to_numpy(), 0.5) for m, c in METHODS.items()}
        for r in FM.score(y, prot, models, args.boot, args.seed):
            if r["metric"] in ("AUPRC", "AUROC"):
                metric_rows.append({"evaluation_set": name, "n_protein": int(prot.max()) + 1,
                                    "n_positive": int(y.sum()), **{k: r[k] for k in ("method", "metric", "value", "sd",
                                                                                      "ci_lo", "ci_hi")}})
        for comp in ("AdaBind", "P2Rank"):
            for r in paired_curve(y, prot, models["AF2BIND"][0], models[comp][0], args.boot, args.seed):
                diff_rows.append({"evaluation_set": name, "comparison": f"AF2BIND - {comp}", **r})
    metrics, diffs = pd.DataFrame(metric_rows), pd.DataFrame(diff_rows)
    metrics.to_csv(dest / "af2bind_label_matched_metrics.csv", index=False)
    diffs.to_csv(dest / "af2bind_label_matched_diff.csv", index=False)
    pd.set_option("display.width", 200)
    print(metrics.round(3).to_string(index=False))
    print(diffs.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
