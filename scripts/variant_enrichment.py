#!/usr/bin/env python
"""Are residues predicted to bind a ligand enriched for disease-causing missense variants?

Human Swiss-Prot proteins of TestSet300 and TestSetNew46 only. Variants: UniProt humsavar (LP/P = likely pathogenic or
pathogenic, LB/B = likely benign or benign; ACMG/AMP categories curated by UniProt). Relative solvent accessibility (RSA):
AlphaFold DB model of the canonical sequence, Shrake-Rupley SASA (Biopython) / maximum ASA of Tien et al. 2013.

Inputs
    reports/finalize_results/predictions/test_residue_predictions.csv.gz    (scripts/finalize_metrics.py)
    data/external/variants/humsavar.txt                                      (UniProt release recorded in the manifest)
Caches written under data/external/: uniprot_test_proteins.tsv, alphafold/AF-<acc>.pdb
Outputs (reports/variant_enrichment/): residues.csv.gz, enrichment.csv, cases.csv, manifest.json

    python scripts/variant_enrichment.py [--boot 2000] [--seed 2026]

Binding call of a method = union over its ligand classes of p >= its decision threshold (development-locked for AdaBind,
0.5 for bindNode24 and bindEmbed21DL). Matched top-k = in each protein the ceil(f * L) residues with the highest
threshold-normalised score max_c p_c / thr_c, so that every method selects the same number of residues.
"""
import argparse
import hashlib
import io
import json
import math
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Align
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from statsmodels.discrete.conditional_models import ConditionalLogit

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"
THR = {"AdaBind": {"metal": 0.9148463010787964, "small": 0.4901010101010101, "dna": 0.5643434343434344,
                   "rna": 0.7771717171717172}}
TASKS = {"AdaBind": ["metal", "small", "dna", "rna"], "bindNode24": ["metal", "small", "nuclear"],
         "bindEmbed21DL": ["metal", "small", "nuclear"]}
METHODS = list(TASKS)
MAX_ASA = {"A": 129, "R": 274, "N": 195, "D": 193, "C": 167, "E": 223, "Q": 225, "G": 104, "H": 224, "I": 197,
           "L": 201, "K": 236, "M": 224, "F": 240, "P": 159, "S": 155, "T": 172, "W": 285, "Y": 263, "V": 174}
AA3 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H",
       "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
       "Tyr": "Y", "Val": "V"}


def get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(3 * (i + 1))


def uniprot_table(ids):
    fp = EXT / "uniprot_test_proteins.tsv"
    if fp.exists():
        t = pd.read_csv(fp, sep="\t")
        if set(ids) <= set(t.accession):
            return t
    rows = []
    for i in range(0, len(ids), 100):
        chunk = ",".join(ids[i:i + 100])
        txt = get(f"https://rest.uniprot.org/uniprotkb/accessions?accessions={chunk}"
                  "&fields=accession,reviewed,organism_id,gene_primary,protein_name,sequence&format=tsv&size=500").decode()
        rows.append(pd.read_csv(io.StringIO(txt), sep="\t"))
    t = pd.concat(rows, ignore_index=True).rename(columns={
        "Entry": "accession", "Reviewed": "reviewed", "Organism (ID)": "taxon", "Gene Names (primary)": "gene",
        "Protein names": "protein", "Sequence": "sequence"})
    EXT.mkdir(parents=True, exist_ok=True)
    t.to_csv(fp, sep="\t", index=False)
    return t


def map_positions(seq_ds, seq_up):
    """Dataset position (1-based) -> canonical UniProt position, identical residues only."""
    off = seq_up.find(seq_ds)
    if off >= 0:
        return {i + 1: off + i + 1 for i in range(len(seq_ds))}
    al = Align.PairwiseAligner(mode="global", open_gap_score=-10, extend_gap_score=-0.5,
                               substitution_matrix=Align.substitution_matrices.load("BLOSUM62"))
    a = al.align(seq_ds, seq_up)[0]
    out = {}
    for (s0, s1), (t0, t1) in zip(*a.aligned):
        for k in range(s1 - s0):
            if seq_ds[s0 + k] == seq_up[t0 + k]:
                out[s0 + k + 1] = t0 + k + 1
    return out


def parse_humsavar(fp, accs):
    rows = []
    pat = re.compile(r"^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$")
    for line in fp.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[1] not in accs:
            continue
        m = pat.match(parts[3])
        if not m or parts[4] not in ("LP/P", "LB/B", "US") or m.group(1) not in AA3 or m.group(3) not in AA3:
            continue
        rows.append({"accession": parts[1], "gene": parts[0], "pos": int(m.group(2)), "ref": AA3[m.group(1)],
                     "alt": AA3[m.group(3)], "category": parts[4], "disease": " ".join(parts[6:])})
    return pd.DataFrame(rows)


def rsa_for(acc):
    d = EXT / "alphafold"
    d.mkdir(parents=True, exist_ok=True)
    fp = d / f"AF-{acc}.pdb"
    if not fp.exists():
        meta = json.loads(get(f"https://alphafold.ebi.ac.uk/api/prediction/{acc}"))
        entry = next(e for e in meta if e.get("uniprotAccession") == acc)
        fp.write_bytes(get(entry["pdbUrl"]))
    st = PDBParser(QUIET=True).get_structure(acc, fp)
    ShrakeRupley().compute(st[0], level="R")
    out = {}
    for res in st[0].get_residues():
        aa = AA3.get(res.get_resname().capitalize())
        if aa:
            out[res.id[1]] = (aa, min(res.sasa / MAX_ASA[aa], 1.0))
    return out


def add_scores(d):
    """Per method: binding call at its threshold (union of classes) and threshold-normalised score."""
    for m in METHODS:
        thr = THR.get(m, {})
        ratio = np.column_stack([d[f"p_{t}__{m}"].to_numpy() / thr.get(t, 0.5) for t in TASKS[m]])
        d[f"score__{m}"] = ratio.max(axis=1)
        d[f"call__{m}"] = (ratio >= 1).any(axis=1)
    return d


def topk(d, frac):
    out = {}
    groups = d.groupby("prot_id", sort=False).indices
    for m in METHODS:
        s = d[f"score__{m}"].to_numpy()
        sel = np.zeros(len(d), bool)
        for idx in groups.values():
            k = max(1, math.ceil(frac * len(idx)))
            sel[idx[np.argsort(-s[idx], kind="mergesort")[:k]]] = True
        out[m] = sel
    return out


def mh_or(x, y, strata):
    """Mantel-Haenszel odds ratio of y (bool) for x (bool) across strata (vectorised)."""
    n = strata.max() + 1
    a = np.bincount(strata, weights=(x & y), minlength=n)
    b = np.bincount(strata, weights=(x & ~y), minlength=n)
    c = np.bincount(strata, weights=(~x & y), minlength=n)
    e = np.bincount(strata, weights=(~x & ~y), minlength=n)
    t = a + b + c + e
    ok = t > 0
    num, den = (a * e / np.where(ok, t, 1))[ok].sum(), (b * c / np.where(ok, t, 1))[ok].sum()
    return num / den if den > 0 else np.nan


def boot_or(x_by, y, strata, B, seed):
    """MH OR per selector, percentile 95% CI over protein resamples, paired log-OR difference AdaBind minus method."""
    rng = np.random.default_rng(seed)
    n_p = strata.max() + 1
    pos = [np.flatnonzero(strata == p) for p in range(n_p)]
    point = {k: mh_or(v, y, strata) for k, v in x_by.items()}
    reps = {k: [] for k in x_by}
    for _ in range(B):
        draw = rng.integers(0, n_p, n_p)
        idx = np.concatenate([pos[p] for p in draw])
        st = np.repeat(np.arange(n_p), [len(pos[p]) for p in draw])
        for k, v in x_by.items():
            reps[k].append(mh_or(v[idx], y[idx], st))
    rows = []
    for k in x_by:
        orr = np.asarray(reps[k], float)
        r = np.log(orr)
        row = {"selector": k, "OR": point[k], "sd": float(np.std(orr[np.isfinite(orr)], ddof=1)),
               "ci_lo": float(np.exp(np.nanpercentile(r, 2.5))), "ci_hi": float(np.exp(np.nanpercentile(r, 97.5)))}
        meth, rule = k.split("|", 1)
        ref = f"AdaBind|{rule}"
        if meth in METHODS and meth != "AdaBind" and ref in reps:
            diff = np.log(np.asarray(reps[ref], float)) - r
            row.update({"logOR_diff_AdaBind_minus": float(np.log(point[ref]) - np.log(point[k])),
                        "diff_sd": float(np.nanstd(diff[np.isfinite(diff)], ddof=1)), "diff_ci_lo": float(np.nanpercentile(diff, 2.5)),
                        "diff_ci_hi": float(np.nanpercentile(diff, 97.5))})
        rows.append(row)
    return rows


def clogit(y, x, rsa, strata):
    """Conditional logistic regression (strata = protein) of y on x, adjusted for RSA; OR of x with 95% CI."""
    keep = pd.Series(y).groupby(strata).transform(lambda s: 0 < s.sum() < len(s)).to_numpy()
    X = np.column_stack([x[keep].astype(float), rsa[keep]])
    res = ConditionalLogit(y[keep].astype(float), X, groups=strata[keep]).fit(disp=False)
    lo, hi = np.asarray(res.conf_int())[0]
    o = float(np.exp(res.params[0]))
    return o, float(np.exp(lo)), float(np.exp(hi)), int(keep.sum()), o * float(res.bse[0])  # SD by the delta method


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", default=str(ROOT / "reports/finalize_results/predictions/test_residue_predictions.csv.gz"))
    ap.add_argument("--humsavar", default=str(EXT / "variants/humsavar.txt"))
    ap.add_argument("--out", default=str(ROOT / "reports/variant_enrichment"))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(args.pred)
    d["prot_id"] = d.prot_id.astype(str)
    seqs = d.groupby("prot_id").aa.apply("".join)
    up = uniprot_table(sorted(seqs.index))
    human = up[(up.taxon == 9606) & (up.reviewed == "reviewed")].drop_duplicates("accession").set_index("accession")
    print(f"test proteins {len(seqs)}; human Swiss-Prot {len(human)}", flush=True)

    hv = parse_humsavar(Path(args.humsavar), set(human.index))
    release = re.search(r"Release:\s+(\S+ of \S+)", Path(args.humsavar).read_text()[:3000]).group(1)
    d = d[d.prot_id.isin(human.index)].reset_index(drop=True)
    pos_map, rsa_map, n_rsa_missing = {}, {}, 0
    for acc in sorted(d.prot_id.unique()):
        mp = map_positions(seqs[acc], human.loc[acc, "sequence"])
        try:
            rsa = rsa_for(acc)
        except Exception as e:  # no AlphaFold DB model for this accession
            print(f"no AlphaFold RSA for {acc}: {e}", file=sys.stderr)
            rsa, n_rsa_missing = {}, n_rsa_missing + 1
        for r_ds, r_up in mp.items():
            pos_map[(acc, r_ds)] = r_up
            if r_up in rsa and rsa[r_up][0] == seqs[acc][r_ds - 1]:
                rsa_map[(acc, r_ds)] = rsa[r_up][1]
    d["uniprot_pos"] = [pos_map.get(k) for k in zip(d.prot_id, d.resid)]
    d["rsa"] = [rsa_map.get(k, np.nan) for k in zip(d.prot_id, d.resid)]

    seq_up = human.sequence.to_dict()
    hv["ref_ok"] = [p <= len(seq_up[a]) and seq_up[a][p - 1] == r for a, p, r in zip(hv.accession, hv.pos, hv.ref)]
    mism = int((~hv.ref_ok).sum())
    hv = hv[hv.ref_ok]
    covered = set(zip(d.prot_id, d.uniprot_pos))
    hv["in_test_sequence"] = [(a, p) in covered for a, p in zip(hv.accession, hv.pos)]
    hv.to_csv(out / "variants_mapped.csv", index=False)
    cat = hv.groupby(["accession", "pos"]).category.agg(set).to_dict()
    key = list(zip(d.prot_id, d.uniprot_pos))
    d["has_P"] = [("LP/P" in cat.get(k, set())) for k in key]
    d["has_B"] = [("LB/B" in cat.get(k, set())) and ("LP/P" not in cat.get(k, set())) for k in key]
    d["y_any"] = (d.y_metal_binding.astype(bool) | d.y_small_binding.astype(bool) | d.y_nuclear_binding.astype(bool))
    d = add_scores(d)

    prot_p = d.groupby("prot_id").has_P.sum()
    keep_p = prot_p[prot_p > 0].index
    print(f"human proteins with >=1 mapped LP/P variant: {len(keep_p)}; LP/P residues {int(d.has_P.sum())}; "
          f"LB/B residues {int(d.has_B.sum())}; humsavar reference mismatches dropped {mism}", flush=True)
    dp = d[d.prot_id.isin(keep_p)].reset_index(drop=True)
    strata = pd.factorize(dp.prot_id)[0]
    yP = dp.has_P.to_numpy()

    sel = {"Annotated|binding sites": dp.y_any.to_numpy()}
    sel.update({f"{m}|threshold": dp[f"call__{m}"].to_numpy() for m in METHODS})
    for f in (0.05, 0.10):
        tk = topk(dp, f)
        sel.update({f"{m}|top {int(f * 100)}%": tk[m] for m in METHODS})

    rows = []
    for r in boot_or(sel, yP, strata, args.boot, args.seed):
        rows.append({"analysis": "LP/P residues among all residues", **r, "n_selected": int(sel[r["selector"]].sum())})
    unl = ~dp.y_any.to_numpy()
    sel_u = {k: v[unl] for k, v in sel.items() if not k.startswith("Annotated")}
    st_u = pd.factorize(strata[unl])[0]
    for r in boot_or(sel_u, yP[unl], st_u, args.boot, args.seed):
        rows.append({"analysis": "LP/P residues among unannotated residues", **r,
                     "n_selected": int(sel_u[r["selector"]].sum())})
    vb = (dp.has_P | dp.has_B).to_numpy()
    sel_pb = {k: v[vb] for k, v in sel.items()}
    st_pb = pd.factorize(strata[vb])[0]
    for r in boot_or(sel_pb, yP[vb], st_pb, args.boot, args.seed):
        rows.append({"analysis": "LP/P versus LB/B variant residues", **r, "n_selected": int(sel_pb[r["selector"]].sum())})
    ok = ~np.isnan(dp.rsa.to_numpy())
    for k, v in sel.items():
        o, lo, hi, n, sd = clogit(yP[ok], v[ok], dp.rsa.to_numpy()[ok], pd.factorize(strata[ok])[0])
        rows.append({"analysis": "LP/P residues, adjusted for RSA", "selector": k, "OR": o, "sd": sd, "ci_lo": lo, "ci_hi": hi,
                     "n_selected": int(v[ok].sum()), "n_residues_model": n})
    res = pd.DataFrame(rows)
    res.to_csv(out / "enrichment.csv", index=False)
    pd.set_option("display.width", 240)
    print(res.round(3).to_string(index=False))

    cases = dp.groupby("prot_id").agg(length=("resid", "size"), lp_p=("has_P", "sum"), lb_b=("has_B", "sum"),
                                      annotated=("y_any", "sum"))
    for m in METHODS:
        cases[f"called_{m}"] = dp.groupby("prot_id")[f"call__{m}"].sum()
        cases[f"lp_p_called_{m}"] = dp[dp[f"call__{m}"] & dp.has_P].groupby("prot_id").size()
    cases = cases.fillna(0).join(human[["gene", "protein"]]).sort_values("lp_p", ascending=False)
    cases.to_csv(out / "cases.csv")
    d.to_csv(out / "residues.csv.gz", index=False)
    (out / "manifest.json").write_text(json.dumps({
        "humsavar_release": release, "humsavar_sha256": hashlib.sha256(Path(args.humsavar).read_bytes()).hexdigest(),
        "n_test_proteins": int(len(seqs)), "n_human_swissprot": int(len(human)),
        "n_proteins_with_lp_p": int(len(keep_p)), "n_lp_p_residues": int(dp.has_P.sum()),
        "n_lb_b_residues": int(dp.has_B.sum()), "humsavar_reference_mismatches_dropped": mism,
        "proteins_without_alphafold": n_rsa_missing, "bootstrap_B": args.boot, "seed": args.seed,
        "thresholds": {**THR, "baselines": 0.5}, "topk_fractions": [0.05, 0.10]}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
