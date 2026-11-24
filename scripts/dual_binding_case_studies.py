#!/usr/bin/env python
"""Reproduce post-hoc, structure-supported illustrations of the dual-label subset.

Run in the AdaBind environment (Biopython, matplotlib; PyMOL for --render):
    conda run -n AdaBind python scripts/dual_binding_case_studies.py --render

Inputs are cached public RCSB mmCIF/entry JSON, BioLiP2 rows and fixed test
predictions. No model is fitted. Structure contacts and benchmark union labels
are kept separate. Top-N is an evaluation quota, not a deployable threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from sklearn.metrics import average_precision_score

from biolip_shared_residue_ligands import mapping
from p2rank_baseline import load_p2rank

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/dual_binding_cases"
STRUCTURES = ROOT / "data/external/case_structures"
PRED = ROOT / "reports/finalize_results/predictions/test_residue_predictions.csv.gz"
CASES = [
    {"prot_id": "O15496", "name": "PLA2G10", "pdb": "4UY1", "chain": "A", "metal": "CA", "ligand": "TJM",
     "ligand_role": "pyrazole inhibitor", "label": "Ca / inhibitor", "doi": "10.1016/j.bmcl.2014.09.058",
     "caveat": "Precursor numbering differs from the mature experimental chain; contacts do not establish functional necessity."},
    {"prot_id": "P22636", "name": "LigAB beta", "pdb": "1B4U", "chain": "B", "metal": "FE", "ligand": "DHB",
     "ligand_role": "protocatechuate substrate", "label": "Fe / protocatechuate", "doi": "10.1016/S0969-2126(99)80122-1",
     "caveat": "The publication describes a nonphysiological Fe(III) trapped complex; the active enzyme uses Fe(II). "
               "An alpha subunit also contributes the structural environment; predictions use one ESMFold chain."},
    {"prot_id": "O50580", "name": "Tagatose epimerase", "pdb": "2QUM", "chain": "A", "metal": "MN", "ligand": "TAG",
     "ligand_role": "D-tagatose substrate", "label": "Mn / D-tagatose", "doi": "10.1016/j.jmb.2007.09.033",
     "caveat": "Illustrative boundary case: recovery of this substrate complex does not imply the best ranking of "
               "all benchmark metal or small-molecule labels."},
]
METHODS = ["AdaBind", "bindNode24", "bindEmbed21DL", "ESMBind", "MetalBind", "P2Rank", "AF2BIND"]
TASK_METHODS = {"metal": METHODS[:5], "small": METHODS[:3] + ["P2Rank", "AF2BIND"]}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atom_frame(path):
    cif = MMCIF2Dict(str(path))
    keys = ["id", "group_PDB", "type_symbol", "label_atom_id", "label_alt_id", "label_comp_id",
            "auth_asym_id", "auth_seq_id", "Cartn_x", "Cartn_y", "Cartn_z", "occupancy", "pdbx_PDB_model_num"]
    a = pd.DataFrame({key: cif["_atom_site." + key] for key in keys})
    for key in ["Cartn_x", "Cartn_y", "Cartn_z", "occupancy"]:
        a[key] = a[key].astype(float)
    a = a[(a.pdbx_PDB_model_num == "1") & ~a.type_symbol.isin(["H", "D"]) & (a.occupancy > 0)].copy()
    a = a.sort_values(["occupancy", "label_alt_id"], ascending=[False, True]).drop_duplicates(
        ["group_PDB", "auth_asym_id", "auth_seq_id", "label_comp_id", "label_atom_id"])
    return cif, a


def coordinates(frame):
    return frame[["Cartn_x", "Cartn_y", "Cartn_z"]].to_numpy()


def nearest(a, b):
    distances = np.linalg.norm(coordinates(a)[:, None] - coordinates(b)[None], axis=-1)
    i, j = np.unravel_index(np.argmin(distances), distances.shape)
    return float(distances[i, j]), a.iloc[i], b.iloc[j]


def top_ranks(values):
    # Deterministic sequence-order tie breaking, shared by all models.
    order = np.argsort(-np.asarray(values), kind="mergesort")
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def analyse():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = json.loads((OUT / "biolip_candidates.json").read_text())
    pred = pd.read_csv(PRED)
    manifest = json.loads((ROOT / "reports/finalize_results/manifest.json").read_text())
    thresholds = manifest["thresholds"]
    curated, contact_rows, metric_rows, residue_rows, exact_rows = [], [], [], [], []
    source_files = {str(PRED.relative_to(ROOT)): digest(PRED),
                    "reports/dual_binding_cases/biolip_candidates.json": digest(OUT / "biolip_candidates.json"),
                    "reports/finalize_results/manifest.json": digest(ROOT / "reports/finalize_results/manifest.json")}
    for spec in CASES:
        pid, pdb, chain = spec["prot_id"], spec["pdb"], spec["chain"]
        d = pred[(pred.prot_id == pid) & (pred.split == "TestSet300")].sort_values("resid").reset_index(drop=True)
        assert len(d) and np.array_equal(d.resid, np.arange(1, len(d) + 1))
        sequence = "".join(d.aa)
        assert d.y_metal_binding.sum() > 0 and d.y_small_binding.sum() > 0
        cif_path, entry_path = STRUCTURES / f"{pdb}.cif", STRUCTURES / f"{pdb}.json"
        cif, atoms = atom_frame(cif_path)
        entry = json.loads(entry_path.read_text())
        source_files[str(cif_path.relative_to(ROOT))] = digest(cif_path)
        source_files[str(entry_path.relative_to(ROOT))] = digest(entry_path)
        sites = [r for r in raw if r[0].upper() == pdb and r[1] == chain and r[17] == pid
                 and r[4] in (spec["metal"], spec["ligand"])]
        assert len(sites) == 2 and float(sites[0][2]) <= 2.5
        contact_sets, ligand_atoms = {}, {}
        for site in sites:
            task = "metal" if site[4] == spec["metal"] else "small"
            mp = mapping(site[20], sequence)
            positions = site[8].split()
            authors = site[7].split()
            assert len(positions) == len(authors)
            target = atoms[(atoms.auth_asym_id == site[5]) & (atoms.auth_seq_id == site[19].strip()) &
                           (atoms.label_comp_id == site[4])]
            assert len(target), (pid, site[4], site[19])
            assert target.occupancy.min() > 0
            ligand_atoms[task] = target
            contact_sets[task] = []
            for original, numbered in zip(authors, positions):
                index = mp[int(numbered[1:])]
                aa = original[0]
                assert sequence[index] == aa == numbered[0], (pid, original, index)
                original_number = original[1:]
                protein_atoms = atoms[(atoms.group_PDB == "ATOM") & (atoms.auth_asym_id == chain) &
                                      (atoms.auth_seq_id == original_number)]
                assert len(protein_atoms), (pid, original)
                distance, donor, ligand_atom = nearest(protein_atoms, target)
                if task == "metal":
                    assert donor.type_symbol in ("N", "O", "S") and distance < 3.2
                row = {"prot_id": pid, "pdb": pdb, "chain": chain, "task": task, "ligand": site[4],
                       "ligand_auth_resid": site[19].strip(), "benchmark_resid": index + 1,
                       "aa": aa, "benchmark_label": aa + str(index + 1), "pdb_resid": original_number,
                       "pdb_label": original, "benchmark_positive": bool(d.loc[index, f"y_{task}_binding"]),
                       "closest_protein_atom": donor.label_atom_id, "closest_ligand_atom": ligand_atom.label_atom_id,
                       "min_heavy_atom_distance_A": distance, "ligand_min_occupancy": float(target.occupancy.min())}
                contact_rows.append(row)
                contact_sets[task].append(row)
        metal_distance, _, _ = nearest(ligand_atoms["metal"], ligand_atoms["small"])
        assert metal_distance < 3.2, (pid, metal_distance)
        d["p_small__P2Rank"] = load_p2rank(ROOT / "data/external/p2rank/out_alphafold", d).to_numpy()
        af = pd.read_csv(ROOT / f"data/external/af2bind/out/{pid}.csv").set_index("resid")
        d["p_small__AF2BIND"] = af.p_bind.reindex(d.resid).to_numpy()
        for path in [ROOT / f"data/external/af2bind/out/{pid}.csv",
                     ROOT / f"data/external/p2rank/out_alphafold/{pid}.pdb_residues.csv"]:
            source_files[str(path.relative_to(ROOT))] = digest(path)
        for task, methods in TASK_METHODS.items():
            y = d[f"y_{task}_binding"].to_numpy(int)
            n = int(y.sum())
            for method in methods:
                values = d[f"p_{task}__{method}"].to_numpy()
                assert np.isfinite(values).all()
                ranks = top_ranks(values)
                threshold = thresholds[task] if method == "AdaBind" else .28 if method == "AF2BIND" else .4586 if method == "P2Rank" else .5
                contacts = contact_sets[task]
                recovered = sum(ranks[r["benchmark_resid"] - 1] <= n for r in contacts)
                metric_rows.append({"prot_id": pid, "task": task, "method": method, "AUPRC": average_precision_score(y, values),
                                    "n_benchmark_positive": n, "top_n_benchmark_tp": int(y[ranks <= n].sum()),
                                    "n_structure_contacts": len(contacts), "structure_contacts_in_top_n": recovered,
                                    "threshold": threshold,
                                    "structure_contacts_at_threshold": sum(values[r["benchmark_resid"] - 1] >= threshold for r in contacts)})
                for r in contacts:
                    index = r["benchmark_resid"] - 1
                    residue_rows.append({**r, "method": method, "probability": float(values[index]),
                                         "rank": int(ranks[index]), "top_n": n, "in_top_n": bool(ranks[index] <= n),
                                         "threshold": threshold, "at_threshold": bool(values[index] >= threshold)})
            d[f"top_n_{task}"] = n
        d.to_csv(OUT / f"{pid}_all_residue_predictions.csv", index=False)
        shared = sorted(set(r["benchmark_resid"] for r in contact_sets["metal"]) &
                        set(r["benchmark_resid"] for r in contact_sets["small"]))
        curated.append({**spec, "resolution_A": float(entry["rcsb_entry_info"]["resolution_combined"][0]),
                        "pdb_url": f"https://www.rcsb.org/structure/{pdb}", "paper_url": "https://doi.org/" + spec["doi"],
                        "metal_ligand_min_distance_A": metal_distance, "shared_benchmark_resids": shared,
                        "contact_sets": contact_sets, "length": len(d)})
        print(pid, pdb, "metal/organic min distance", round(metal_distance, 3), "shared", shared, flush=True)
    metrics, residues, contacts = pd.DataFrame(metric_rows), pd.DataFrame(residue_rows), pd.DataFrame(contact_rows)
    metrics.to_csv(OUT / "case_metrics.csv", index=False)
    residues.to_csv(OUT / "contact_predictions.csv", index=False)
    contacts.to_csv(OUT / "verified_contacts.csv", index=False)
    (OUT / "curation.json").write_text(json.dumps(curated, indent=2) + "\n")
    # Record paired wins AND paired misses on the same complete contact set.
    for keys, group in residues.groupby(["prot_id", "task", "benchmark_resid"]):
        group = group.set_index("method")
        for method in group.index.drop("AdaBind"):
            a, b = group.loc["AdaBind"], group.loc[method]
            if bool(a.in_top_n) != bool(b.in_top_n):
                exact_rows.append({"prot_id": keys[0], "task": keys[1], "residue": a.benchmark_label,
                                   "comparator": method, "AdaBind_rank": a["rank"], "comparator_rank": b["rank"],
                                   "top_n": a.top_n, "direction": "AdaBind_only" if a.in_top_n else "comparator_only"})
    pd.DataFrame(exact_rows).to_csv(OUT / "discordant_contacts.csv", index=False)
    pd.DataFrame([{**r, "contact_sets": None} for r in curated]).drop(columns="contact_sets").to_csv(OUT / "case_selection.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps({"script_sha256": digest(Path(__file__)), "input_sha256": source_files,
        "selection": "post-hoc illustrative cases covering Ca/inhibitor, Fe/substrate and Mn/substrate; not an independent validation cohort",
        "ranking": "descending probability, stable sequence-order ties; N is the number of benchmark positives for that task",
        "ablation_case_comparison": "unavailable: no per-residue ablation test predictions or checkpoints in the local input caches",
        "no_new_training": True, "no_case_level_significance_tests": True}, indent=2) + "\n")
    return curated, metrics, residues


def render(curated):
    import pymol2
    with pymol2.PyMOL() as session:
        cmd = session.cmd
        for case in curated:
            cmd.reinitialize()
            cmd.set("cif_use_auth", 1)
            cmd.load(str(STRUCTURES / (case["pdb"] + ".cif")), "complex")
            cmd.remove("hydro")
            cmd.hide("everything")
            cmd.bg_color("white")
            chain = case["chain"]
            cmd.select("case_receptor", f"complex and polymer.protein and chain {chain}")
            metal_site, ligand_site = case["contact_sets"]["metal"][0], case["contact_sets"]["small"][0]
            cmd.select("case_ion", f"complex and chain {chain} and resn {case['metal']} and resi {metal_site['ligand_auth_resid']}")
            cmd.select("case_ligand", f"complex and chain {chain} and resn {case['ligand']} and resi {ligand_site['ligand_auth_resid']}")
            assert cmd.count_atoms("case_ion") and cmd.count_atoms("case_ligand")
            author_resids = "+".join(r["pdb_resid"] for r in case["contact_sets"]["metal"])
            cmd.select("case_donors", f"case_receptor and resi {author_resids}")
            cmd.show("cartoon", "case_receptor")
            cmd.color("gray80", "case_receptor")
            cmd.set("cartoon_transparency", .65)
            cmd.show("sticks", "case_donors or case_ligand")
            cmd.set_color("contact_carbon", [.90, .60, .0])
            cmd.set_color("ligand_carbon", [.0, .62, .45])
            cmd.color("contact_carbon", "case_donors and elem C")
            cmd.color("ligand_carbon", "case_ligand and elem C")
            cmd.color("red", "(case_donors or case_ligand) and elem O")
            cmd.color("blue", "(case_donors or case_ligand) and elem N")
            cmd.show("spheres", "case_ion")
            cmd.color("violet", "case_ion")
            cmd.set("sphere_scale", .40, "case_ion")
            cmd.set("stick_radius", .16)
            for r in case["contact_sets"]["metal"]:
                cmd.label(f"case_donors and resi {r['pdb_resid']} and name CA", json.dumps(r["benchmark_label"]))
            cmd.set("label_color", "black")
            cmd.set("label_size", 32)
            cmd.set("label_outline_color", "white")
            cmd.set("orthoscopic", 1)
            cmd.set("depth_cue", 0)
            cmd.set("specular", .1)
            cmd.set("ray_opaque_background", 0)
            cmd.set("antialias", 2)
            cmd.orient("case_donors or case_ligand or case_ion")
            cmd.zoom("case_donors or case_ligand or case_ion", buffer=5)
            cmd.png(str(OUT / f"{case['prot_id']}_structure.png"), width=1200, height=1000, dpi=600, ray=1)


def figure(curated, metrics, residues):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 7.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(7.0, 8.6))
    grid = fig.add_gridspec(3, 3, width_ratios=[1.05, 1.12, 1.34], wspace=.56, hspace=.64,
                           left=.025, right=.985, top=.92, bottom=.13)
    for row, case in enumerate(curated):
        pid = case["prot_id"]
        ax = fig.add_subplot(grid[row, 0])
        ax.imshow(plt.imread(OUT / f"{pid}_structure.png"))
        ax.axis("off")
        ax.set_title(case["name"] + "\n" + pid + " | " + case["pdb"] + " " + case["chain"], fontsize=7.5)
        ax.text(.5, -.03, case["label"] + f"\n{case['resolution_A']:.2f} Å experimental structure",
                ha="center", va="top", transform=ax.transAxes, fontsize=7)
        ax.text(-.03, 1.12, chr(ord("a") + row * 3), transform=ax.transAxes, weight="bold", fontsize=10)
        plot = fig.add_subplot(grid[row, 1])
        for task, color, marker, offset in [("metal", "#D55E00", "o", -.12), ("small", "#0072B2", "s", .12)]:
            q = metrics[(metrics.prot_id == pid) & (metrics.task == task)].set_index("method")
            for j, method in enumerate(METHODS):
                if method in q.index:
                    plot.scatter(q.loc[method, "AUPRC"], j + offset, color=color, marker=marker, s=18, zorder=3)
        plot.set_yticks(np.arange(len(METHODS)), METHODS)
        plot.set_ylim(len(METHODS) - .5, -.5)
        plot.set_xlim(0, 1.02)
        plot.set_xticks([0, .5, 1])
        plot.set_xlabel("AUPRC, benchmark union labels")
        plot.set_title("Whole-protein ranking", fontsize=7.5)
        plot.spines[["top", "right"]].set_visible(False)
        plot.grid(axis="x", color=".9", lw=.5)
        plot.text(-.43, 1.10, chr(ord("a") + row * 3 + 1), transform=plot.transAxes, weight="bold", fontsize=10)
        tile = fig.add_subplot(grid[row, 2])
        columns = [(task, r) for task in ("metal", "small") for r in case["contact_sets"][task]]
        values = np.full((len(METHODS), len(columns)), -1.0)
        for col, (task, contact) in enumerate(columns):
            q = residues[(residues.prot_id == pid) & (residues.task == task) &
                         (residues.benchmark_resid == contact["benchmark_resid"])].set_index("method")
            for j, method in enumerate(METHODS):
                if method in q.index:
                    values[j, col] = int(q.loc[method, "in_top_n"])
        cmap = ListedColormap(["#E6E6E6", "#FFFFFF", "#333333"])
        tile.imshow(values, cmap=cmap, norm=BoundaryNorm([-1.5, -.5, .5, 1.5], 3), aspect="auto")
        for j in range(len(METHODS)):
            for col in range(len(columns)):
                if values[j, col] == 0:
                    tile.text(col, j, "×", ha="center", va="center", color=".55", fontsize=7)
                elif values[j, col] < 0:
                    tile.text(col, j, "–", ha="center", va="center", color=".55", fontsize=6)
        tile.set_xticks(np.arange(len(columns)), [r["benchmark_label"] for _, r in columns], rotation=90)
        tile.set_yticks(np.arange(len(METHODS)), [""] * len(METHODS))
        tile.tick_params(axis="both", length=0)
        tile.set_xticks(np.arange(-.5, len(columns), 1), minor=True)
        tile.set_yticks(np.arange(-.5, len(METHODS), 1), minor=True)
        tile.grid(which="minor", color=".8", lw=.45)
        tile.tick_params(which="minor", length=0)
        nm = len(case["contact_sets"]["metal"])
        tile.axvline(nm - .5, color="black", lw=.9)
        tile.text((nm - 1) / 2, -.82, "Metal", ha="center", color="#D55E00", fontsize=7)
        tile.text((nm + len(columns) - 1) / 2, -.82, "Small molecule", ha="center", color="#0072B2", fontsize=7)
        tile.set_title("Structure contacts in top N", fontsize=7.5, pad=20)
        for spine in tile.spines.values():
            spine.set_visible(False)
        tile.text(-.11, 1.10, chr(ord("a") + row * 3 + 2), transform=tile.transAxes, weight="bold", fontsize=10)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color="#D55E00", label="Metal-ion AUPRC"),
                        Line2D([], [], marker="s", ls="", color="#0072B2", label="Small-molecule AUPRC")],
               loc="lower left", bbox_to_anchor=(.04, .035), ncol=2, frameon=False, fontsize=7)
    fig.text(.54, .055, "■ Recovered   × Missed   – Task not applicable", fontsize=7)
    fig.text(.04, .015, "N = benchmark binding residues per protein and task. Descriptive, post-hoc cases; no case-level significance tests.", fontsize=7)
    fig.savefig(OUT / "fig_dual_binding_cases.pdf")
    fig.savefig(OUT / "fig_dual_binding_cases.png", dpi=450)
    plt.close(fig)


def report(curated, metrics, residues):
    lines = ["# Structure-supported dual-label case studies", "",
             "Three post-hoc illustrations, not new experimental validation or an independent benchmark. "
             "Whole-protein AUPRC uses the original union labels; contact recovery uses a named experimental complex. "
             "No ablated-model residue predictions were available locally, so no component-level attribution is made.", ""]
    for case in curated:
        pid = case["prot_id"]
        lines += ["## " + case["name"] + " (" + pid + ")", "",
                  f"Experimental complex: [{case['pdb']} {case['chain']}]({case['pdb_url']}), "
                  f"{case['resolution_A']:.2f} Å; {case['metal']} and {case['ligand']} ({case['ligand_role']}). "
                  f"[Primary publication]({case['paper_url']}).", "",
                  f"Minimum metal–organic heavy-atom distance: {case['metal_ligand_min_distance_A']:.2f} Å. "
                  "Shared structure-contact residues (benchmark numbering): " + ", ".join(map(str, case["shared_benchmark_resids"])) + ".", "",
                  "| Method | Metal AUPRC | Small AUPRC | Metal contacts in top N | Small contacts in top N |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for method in METHODS:
            cells = []
            for task in ("metal", "small"):
                q = metrics[(metrics.prot_id == pid) & (metrics.task == task) & (metrics.method == method)]
                cells.append(q.iloc[0] if len(q) else None)
            lines.append("| " + method + " | " + " | ".join(["—" if q is None else f"{q.AUPRC:.3f}" for q in cells] +
                         ["—" if q is None else f"{q.structure_contacts_in_top_n}/{q.n_structure_contacts}" for q in cells]) + " |")
        lines += ["", "Caveat: " + case["caveat"], ""]
        lines += ["Frozen-threshold contact recovery (distinct from top-N retrieval):", "",
                  "| Method | Task | Threshold | Structure contacts called |",
                  "| --- | --- | ---: | ---: |"]
        for q in metrics[metrics.prot_id == pid].itertuples():
            lines.append(f"| {q.method} | {q.task} | {q.threshold:.4f} | {q.structure_contacts_at_threshold}/{q.n_structure_contacts} |")
        lines.append("")
    lines += ["## Reproducibility and interpretation", "",
              "- All residue positions in figures use benchmark sequence numbering; verified_contacts.csv records the experimental author numbering and atom distances.",
              "- The quota N is the number of benchmark positives for that protein/task, shared by every method; ties are broken by sequence order. It is not an inference-time threshold.",
              "- contact_predictions.csv also reports the frozen AdaBind thresholds, baseline 0.5 thresholds, P2Rank development threshold and AF2BIND published 0.28 reference threshold.",
              "- discordant_contacts.csv records both AdaBind-only and comparator-only recoveries; no contacts were selected for display based on model performance.",
              "- Structure views show experimental coordinates; structure-dependent predictions used ESMFold models. Neither coordination chemistry nor catalytic activity is inferred from model scores.",
              "- These cases were chosen after inspection for biological diversity and an informative limitation case. They are not an unbiased sample for estimating a population advantage."]
    (OUT / "case_report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    curated, metrics, residues = analyse()
    if args.render:
        render(curated)
    figure(curated, metrics, residues)
    report(curated, metrics, residues)
    print(metrics[["prot_id", "task", "method", "AUPRC", "structure_contacts_in_top_n", "n_structure_contacts"]].to_string(index=False))


if __name__ == "__main__":
    main()
