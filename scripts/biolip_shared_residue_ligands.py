#!/usr/bin/env python
"""Which ligands make a residue carry both the metal-binding and the small-molecule-binding label?

For every test protein with at least one residue labelled both metal- and small-molecule-binding, read the BioLiP2 sites of
that UniProt entry (resolution <= 2.5 A, as for the benchmark labels), map the receptor sequence onto the benchmark sequence,
and list the ligands that contact each shared residue. Small-molecule ligands are classified as metal-containing (for example
heme or chlorophyll) from the formula in the RCSB Chemical Component Dictionary.

    python scripts/biolip_shared_residue_ligands.py
Inputs: data/external/biolip2/BioLiP.txt.gz, data/zenodo_adabind_data_v1/labels/test*.csv
Outputs: reports/metal_small_colocalisation/{shared_residue_ligands,shared_residue_summary}.csv
"""
import ast
import gzip
import json
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Align

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "zenodo_adabind_data_v1" / "labels"
BIOLIP = ROOT / "data" / "external" / "biolip2" / "BioLiP.txt.gz"
OUT = ROOT / "reports" / "metal_small_colocalisation"
CCD_CACHE = ROOT / "data" / "external" / "biolip2" / "ccd_formulas.json"
# Metal ions as treated by BioLiP (single-atom metal ligands).
METAL_IONS = {"ZN", "CA", "MG", "MN", "FE", "FE2", "CU", "CU1", "CU3", "CO", "NI", "CD", "HG", "NA", "K", "SR", "BA",
              "PB", "PT", "AU", "AG", "MO", "W", "LI", "RB", "CS", "AL", "GA", "IN", "TL", "PD", "IR", "OS", "RU", "RH",
              "V", "CR", "SM", "EU", "GD", "TB", "YB", "LU", "HO", "ER", "DY", "LA", "CE", "PR", "ND", "Y1", "3CO",
              "3NI", "MN3", "FE3", "ZN2", "CO3"}
METAL_ELEMENTS = {"Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
                  "Cu", "Zn", "Ga", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "La", "Ce", "Pr",
                  "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
                  "Au", "Hg", "Tl", "Pb", "Bi", "Al"}


def formula(ligand, cache):
    if ligand not in cache:
        for i in range(3):
            try:
                with urllib.request.urlopen(f"https://data.rcsb.org/rest/v1/core/chemcomp/{ligand}", timeout=60) as r:
                    cache[ligand] = json.loads(r.read())["chem_comp"].get("formula") or ""
                break
            except Exception:
                time.sleep(2 * (i + 1))
                cache[ligand] = cache.get(ligand, "")
    return cache[ligand]


def has_metal(f):
    return any(el in METAL_ELEMENTS for el in re.findall(r"([A-Z][a-z]?)\d*", f or ""))


def is_organic(f):
    """Contains carbon and no metal: excludes inorganic ions such as sulfate or phosphate."""
    return "C" in re.findall(r"([A-Z][a-z]?)\d*", f or "") and not has_metal(f)


def mapping(receptor, seq):
    off = seq.find(receptor)
    if off >= 0:
        return {i + 1: off + i for i in range(len(receptor))}
    al = Align.PairwiseAligner(mode="global", open_gap_score=-10, extend_gap_score=-0.5,
                               substitution_matrix=Align.substitution_matrices.load("BLOSUM62"))
    a = al.align(receptor, seq)[0]
    out = {}
    for (s0, s1), (t0, t1) in zip(*a.aligned):
        for k in range(s1 - s0):
            if receptor[s0 + k] == seq[t0 + k]:
                out[s0 + k + 1] = t0 + k
    return out


def main():
    lab = pd.concat([pd.read_csv(DATA / f) for f in ("testset300_binding_sites.csv", "testsetnew46_binding_sites.csv")])
    targets = {}
    for r in lab.itertuples():
        m = np.flatnonzero(np.array(ast.literal_eval(r.metal_binding)))
        s = np.flatnonzero(np.array(ast.literal_eval(r.small_binding)))
        shared = np.intersect1d(m, s)
        if len(shared):
            targets[r.prot_id] = (r.sequence, set(shared.tolist()))
    rows = []
    with gzip.open(BIOLIP, "rt") as fh:
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if len(c) < 21 or c[17] not in targets:
                continue
            try:
                if float(c[2]) > 2.5:
                    continue
            except ValueError:
                continue
            rows.append({"prot_id": c[17], "pdb": c[0], "chain": c[1], "ligand": c[4], "renum": c[8], "receptor": c[20]})
    sites = pd.DataFrame(rows)
    cache = json.loads(CCD_CACHE.read_text()) if CCD_CACHE.exists() else {}
    out = []
    for pid, (seq, shared) in targets.items():
        sub = sites[sites.prot_id == pid]
        contacts = {i: {"metal": set(), "small": set()} for i in shared}
        same_structure = {i: [] for i in shared}
        for (pdb, chain, receptor), grp in sub.groupby(["pdb", "chain", "receptor"]):
            mp = mapping(receptor, seq)
            local = {i: {"metal": set(), "small": set()} for i in shared}
            for site in grp.itertuples():
                kind = "metal" if site.ligand in METAL_IONS else "small"
                if kind == "small" and site.ligand in {"DNA", "RNA", "rna", "dna"}:
                    continue
                for tok in site.renum.split():
                    k = int(re.sub(r"^[A-Za-z]+", "", tok)) if re.sub(r"^[A-Za-z]+", "", tok).isdigit() else None
                    if k is not None and mp.get(k) in contacts:
                        contacts[mp[k]][kind].add(site.ligand)
                        local[mp[k]][kind].add(site.ligand)
            for i, cc in local.items():
                organic = [l for l in cc["small"] if is_organic(formula(l, cache))]
                if cc["metal"] and organic:
                    same_structure[i].append(f"{pdb}{chain}")
        for i, cc in contacts.items():
            small_metal = sorted(l for l in cc["small"] if has_metal(formula(l, cache)))
            small_organic = sorted(l for l in cc["small"] if is_organic(formula(l, cache)))
            small_inorganic = sorted(l for l in cc["small"] if l not in small_metal and l not in small_organic)
            if not cc["metal"] and not cc["small"]:
                cat = "not found in BioLiP2 (resolution <= 2.5 A)"
            elif small_metal and not small_organic:
                cat = "metal-containing cofactor only"
            elif small_organic and cc["metal"]:
                cat = "metal ion and organic ligand" + (" (plus metal-containing cofactor)" if small_metal else "")
            elif small_organic:
                cat = "organic ligand; no metal ion row found"
            elif small_inorganic and cc["metal"]:
                cat = "metal ion and inorganic ion only"
            elif small_inorganic:
                cat = "inorganic ion only"
            else:
                cat = "metal ion only"
            out.append({"prot_id": pid, "residue_index0": i, "aa": seq[i], "metal_ions": " ".join(sorted(cc["metal"])),
                        "metal_containing_small": " ".join(small_metal), "organic_small": " ".join(small_organic),
                        "inorganic_small": " ".join(small_inorganic),
                        "category": cat, "metal_and_organic_same_structure": bool(same_structure[i]),
                        "example_structures": " ".join(same_structure[i][:3])})
    CCD_CACHE.write_text(json.dumps(cache, indent=1, sort_keys=True))
    res = pd.DataFrame(out)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "shared_residue_ligands.csv", index=False)
    prot_cat = res.groupby("prot_id").category.agg(lambda s: "; ".join(sorted(set(s))))
    summ = pd.DataFrame({"residues": res.category.value_counts(),
                         "proteins_with_category": res.groupby("category").prot_id.nunique()})
    summ.to_csv(OUT / "shared_residue_summary.csv")
    pd.set_option("display.width", 220)
    print(f"proteins with shared residues: {len(targets)}; shared residues: {len(res)}")
    print(summ.to_string())
    print(res.aa.value_counts().head(8).to_string())
    ss = res[res.metal_and_organic_same_structure]
    print(f"residues contacting a metal ion and an organic ligand in the same structure: {len(ss)} "
          f"in {ss.prot_id.nunique()} proteins")
    print(res.groupby("category").apply(lambda g: " ".join(sorted(set(" ".join(g.organic_small + " " + g.metal_containing_small).split())))[:300]).to_string())


if __name__ == "__main__":
    main()
