#!/usr/bin/env python
"""Audit saved predictions without fitting a model or tuning on a test label.

Report threshold metrics under the original inclusion rule and two fixed protein
sets. AF2BIND's published 0.28 threshold is a sensitivity comparison, not a
threshold calibrated on AdaBind's development set. Missing development predictions
and ablation checkpoints are recorded rather than reconstructed from test scores.

Run: OPENBLAS_NUM_THREADS=1 python scripts/audit_operating_points.py --boot 10000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score

import finalize_metrics as FM
from p2rank_baseline import load_p2rank
from p2rank_threshold_metrics import dev_frame, thresholds_module

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "reports/finalize_results/predictions/test_residue_predictions.csv.gz"
MANIFEST = ROOT / "reports/finalize_results/manifest.json"
AF_DIR = ROOT / "data/external/af2bind/out"
P2_DIR = ROOT / "data/external/p2rank/out_alphafold"
P2_DEV = ROOT / "data/external/p2rank/out_alphafold_dev"
METRICS = ("precision", "recall", "F1", "MCC")
RULES = ("manuscript_inclusion", "fixed_all_proteins", "fixed_annotated_positive_proteins")
TASKS = {
    "metal": ("y_metal_binding", "p_metal", ["ESMBind", "bindNode24", "bindEmbed21DL", "MetalBind"]),
    "small": ("y_small_binding", "p_small", ["bindNode24", "bindEmbed21DL"]),
    "dna": ("y_dna", "p_nuclear", ["bindNode24", "bindEmbed21DL"]),
    "rna": ("y_rna", "p_nuclear", ["bindNode24", "bindEmbed21DL"]),
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def markdown_table(frame):
    def cell(value):
        return f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
    lines = ["| " + " | ".join(frame.columns) + " |",
             "| " + " | ".join("---" for _ in frame.columns) + " |"]
    lines += ["| " + " | ".join(cell(value) for value in row) + " |"
              for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def bootstrap_counts(n, boot, seed):
    draws = np.random.default_rng(seed).integers(0, n, size=(boot, n))
    return np.asarray([np.bincount(row, minlength=n) for row in draws], dtype=float)


def summary(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return {"sd": float(a.std(ddof=1)), "ci_lo": float(np.percentile(a, 2.5)),
            "ci_hi": float(np.percentile(a, 97.5)), "n_boot": len(a)}


def pooled_metrics(counts):
    tp, fp, fn, tn = np.moveaxis(np.asarray(counts, dtype=float), -1, 0)
    def divide(a, b):
        return np.divide(a, b, out=np.zeros_like(np.asarray(a, dtype=float)), where=b > 0)
    return np.stack([divide(tp, tp + fp), divide(tp, tp + fn),
                     divide(2 * tp, 2 * tp + fp + fn),
                     divide(tp * tn - fp * fn,
                            np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))], axis=-1)


def score_thresholds(y, prot, models, boot, seed):
    """Curve point estimates and paired, protein-bootstrap threshold metrics."""
    n = int(prot.max()) + 1
    positive = np.bincount(prot, weights=y, minlength=n)
    weights = bootstrap_counts(n, boot, seed)
    weights = weights[(weights @ positive) > 0]
    point_rows, diff_rows = [], []
    stored = {}
    for variant, (method, p, threshold, source) in models.items():
        assert len(p) == len(y) and np.isfinite(p).all()
        assert ((p >= 0) & (p <= 1)).all()
        counts = FM.protein_counts(y, p >= threshold, prot, n)
        for rule in RULES:
            eligible = (counts["inc"] if rule == "manuscript_inclusion" else
                        np.ones(n) if rule == "fixed_all_proteins" else (positive > 0).astype(float))
            denominator = weights @ eligible
            points = {k: float(np.sum(counts[k] * eligible) / eligible.sum()) for k in METRICS}
            matrix = np.column_stack([counts[k] * eligible for k in METRICS])
            numerator = weights @ matrix
            reps = np.divide(numerator, denominator[:, None],
                             out=np.full_like(numerator, np.nan), where=denominator[:, None] > 0)
            stored[variant, rule] = (points, reps)
            for j, metric in enumerate(METRICS):
                point_rows.append({"variant": variant, "method": method, "threshold": threshold,
                                   "threshold_source": source, "protein_rule": rule, "metric": metric,
                                   "value": points[metric], "n_proteins": n,
                                   "n_protein_included": int(eligible.sum()), **summary(reps[:, j])})
        # Rank metrics always use all residues; they are not protein-mean metrics.
        for metric, value in [("AUPRC", average_precision_score(y, p)),
                              ("AUROC", roc_auc_score(y, p))]:
            point_rows.append({"variant": variant, "method": method, "threshold": threshold,
                               "threshold_source": source, "protein_rule": "pooled_residues", "metric": metric,
                               "value": float(value), "n_proteins": n,
                               "n_protein_included": n})
        calls = p >= threshold
        c = np.column_stack([np.bincount(prot, weights=mask, minlength=n) for mask in
                             [(y == 1) & calls, (y == 0) & calls,
                              (y == 1) & ~calls, (y == 0) & ~calls]])
        point, reps = pooled_metrics(c.sum(axis=0)), pooled_metrics(weights @ c)
        assert np.allclose(point[2:], [f1_score(y, calls), matthews_corrcoef(y, calls)], atol=1e-12)
        stored[variant, "pooled_residue_counts"] = (dict(zip(METRICS, point)), reps)
        for j, metric in enumerate(METRICS):
            point_rows.append({"variant": variant, "method": method, "threshold": threshold,
                               "threshold_source": source, "protein_rule": "pooled_residue_counts",
                               "metric": metric, "value": float(point[j]), "n_proteins": n,
                               "n_protein_included": n, **summary(reps[:, j])})
    for variant in models:
        if variant == "AdaBind":
            continue
        for rule in (*RULES, "pooled_residue_counts"):
            ap, ar = stored["AdaBind", rule]
            bp, br = stored[variant, rule]
            for j, metric in enumerate(METRICS):
                diff_rows.append({"variant": variant, "protein_rule": rule, "metric": metric,
                                  "diff_AdaBind_minus": ap[metric] - bp[metric],
                                  **summary(ar[:, j] - br[:, j])})
    # Verify the vectorised bootstrap against the manuscript's scalar function.
    for variant, (_, p, threshold, _) in models.items():
        counts = FM.protein_counts(y, p >= threshold, prot, n)
        _, reps = stored[variant, "manuscript_inclusion"]
        for i in range(min(5, len(weights))):
            check = FM.protein_means(counts, weights[i])
            assert np.allclose(reps[i], [check[k] for k in METRICS], rtol=0, atol=1e-12)
    return point_rows, diff_rows


def load_af(pred):
    rows = []
    for pid, group in pred.groupby("prot_id", sort=False):
        a = pd.read_csv(AF_DIR / f"{pid}.csv")
        assert not a.resid.duplicated().any(), pid
        mapping = a.set_index("resid").p_bind
        values = mapping.reindex(group.resid).to_numpy()
        assert np.isfinite(values).all(), pid
        rows.append(pd.Series(values, index=group.index))
    return pd.concat(rows).sort_index()


def ablation_inventory(out):
    folds = pd.read_csv(ROOT / "reports/finalize_results/metrics/table5_foldmean.csv")
    pairs = []
    for task in folds.task.unique():
        d = folds[folds.task == task].set_index("config")
        a, b = d.loc["V1_s0"], d.loc["V1+ESMIF_EF_s0"]
        differences = [float(b[f"fold{k}"] - a[f"fold{k}"]) for k in range(1, 6)]
        assert np.allclose(differences, [d.loc["difference", f"fold{k}"] for k in range(1, 6)])
        pairs.append({"task": task, "comparison": "add_ESMIF_to_existing_hypergraph_seed0",
                      "without": a.fold_mean, "with": b.fold_mean,
                      "mean_fold_difference": float(np.mean(differences)),
                      "sd_fold_difference": float(np.std(differences, ddof=1)),
                      "positive_folds": int(np.sum(np.asarray(differences) > 0)),
                      **{f"difference_fold{k}": v for k, v in enumerate(differences, 1)},
                      "status": "existing_fold_summary_only_not_a_new_training_run"})
    pd.DataFrame(pairs).to_csv(out / "existing_ablation_pairs.csv", index=False)
    inventory = [
        ("metal", "coordination descriptors on fixed ESM2+ESMIF backbone", "manuscript aggregate available; per-seed predictions absent"),
        ("small", "ESM2+ESMIF with/without residue descriptors and candidate context", "full controlled factorial absent; requires training"),
        ("dna_rna", "label-specific controlled descriptor/context ablations", "merged-label summaries are not separate DNA/RNA ablations; requires training"),
        ("small_dna_rna", "matched-size random candidate groups", "not available; requires training"),
        ("all", "three-seed variability and subset66 ablation scoring", "ablation test predictions/checkpoints absent locally"),
    ]
    pd.DataFrame(inventory, columns=["task", "required_comparison", "status"]).to_csv(out / "ablation_inventory.csv", index=False)
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/operating_point_audit")
    args = parser.parse_args()
    started = time.monotonic()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(PRED)
    assert not pred.duplicated(["split", "prot_id", "resid"]).any()
    pred["p_af"] = load_af(pred)
    pred["p_p2"] = load_p2rank(P2_DIR, pred)
    locked = json.loads(MANIFEST.read_text())["thresholds"]
    print("Loaded test predictions and both external baselines", flush=True)

    thresholds = {}
    threshold_rows = []
    for task, column, criterion in [("small", "small_binding", "dev_oof_mcc_opt"),
                                    ("metal", "metal_binding", "dev_oof_positive_f1_opt")]:
        dev = dev_frame(P2_DEV, column)
        assert dev.prot_id.nunique() == 1010
        assert not set(dev.prot_id) & set(pred.prot_id)
        probabilities = load_p2rank(P2_DEV, dev)
        threshold = thresholds_module().optimize_thresholds_exact(dev.y.to_numpy(), probabilities.to_numpy())[criterion]
        thresholds[task] = threshold
        threshold_rows.append({"method": "P2Rank", "task": task, "threshold": threshold,
                               "source": "recomputed_from_DevSet1010_only", "criterion": criterion,
                               "n_development_proteins": 1010, "n_development_residues": len(dev)})
    print("Recomputed P2Rank development thresholds", thresholds, flush=True)
    records, comparisons = [], []
    groups = list(pred.groupby("split", sort=False))
    t = pred[pred.split == "TestSet300"]
    positive = t.groupby("prot_id")[["y_metal_binding", "y_small_binding"]].sum()
    ids = positive[(positive.y_metal_binding > 0) & (positive.y_small_binding > 0)].index
    assert len(ids) == 66
    groups.append(("TestSet300_subset66", t[t.prot_id.isin(ids)]))
    for split, frame in groups:
        for task, (label, baseline_prefix, methods) in TASKS.items():
            if split.endswith("subset66") and task not in ("metal", "small"):
                continue
            d = frame.copy()
            if task in ("dna", "rna"):
                d = d[d.na_kind_known.isin([True, 1, "True", "true"])].copy()
            y = d[label].to_numpy(dtype=int)
            prot = pd.factorize(d.prot_id)[0]
            models = {"AdaBind": ("AdaBind", d[f"p_{task}__AdaBind"].to_numpy(), locked[task], "locked_development_threshold")}
            for method in methods:
                models[method] = (method, d[f"{baseline_prefix}__{method}"].to_numpy(), 0.5, "existing_manuscript_fixed_0.5")
            if task in thresholds:
                models["P2Rank_dev"] = ("P2Rank", d.p_p2.to_numpy(), thresholds[task], "DevSet1010_only")
            if task == "small":
                for variant, threshold, source in [
                    ("AF2BIND_manuscript", locked["small"], "previous_AdaBind_threshold_for_reference"),
                    ("AF2BIND_published", 0.28, "published_validation_threshold_not_DevSet1010_calibrated"),
                ]:
                    models[variant] = ("AF2BIND", d.p_af.to_numpy(), threshold, source)
            points, diffs = score_thresholds(y, prot, models, args.boot, args.seed)
            records.extend({"split": split, "task": task, **r} for r in points)
            comparisons.extend({"split": split, "task": task, **r} for r in diffs)
            print("Scored", split, task, "proteins", len(np.unique(prot)), flush=True)
    metrics = pd.DataFrame(records)
    diffs = pd.DataFrame(comparisons)
    # Reproduce the existing main-table point values before interpreting sensitivity.
    published = pd.read_csv(ROOT / "reports/finalize_results/metrics/table34_mean_sd.csv")
    for r in metrics[metrics.protein_rule.isin(["manuscript_inclusion", "pooled_residues"]) &
                     ~metrics.split.str.endswith("subset66") &
                     metrics.variant.isin(["AdaBind", "bindNode24", "bindEmbed21DL", "ESMBind", "MetalBind"])].itertuples():
        method = {"AdaBind": "ours", "ESMBind": "ESMBind_7ion_max_TRANSFER", "MetalBind": "MetalBind_13ion_max_TRANSFER"}.get(r.method, r.method)
        reference = published[(published.split == r.split) & (published.task == r.task) &
                              (published.method == method) & (published.metric == r.metric)]
        assert len(reference) == 1 and abs(float(reference.value.iloc[0]) - r.value) < 1e-9, r
        if args.boot == 10000 and args.seed == 2026 and r.metric in METRICS:
            for field in ("sd", "ci_lo", "ci_hi"):
                assert abs(float(reference[field].iloc[0]) - getattr(r, field)) < 1e-9, (r, field)
    # Verify the previously evaluated AF2BIND and P2Rank operating points too.
    external = pd.read_csv(ROOT / "reports/p2rank/af2bind_small_metrics.csv")
    for r in metrics[(metrics.task == "small") & ~metrics.split.str.endswith("subset66") &
                     metrics.protein_rule.isin(["manuscript_inclusion", "pooled_residues"]) &
                     metrics.variant.isin(["AF2BIND_manuscript", "P2Rank_dev"])].itertuples():
        reference = external[(external.split == r.split) & (external.method == r.method) &
                             (external.metric == r.metric)]
        assert len(reference) == 1 and abs(float(reference.value.iloc[0]) - r.value) < 1e-9, r
        if args.boot == 10000 and args.seed == 2026 and r.metric in METRICS:
            for field in ("sd", "ci_lo", "ci_hi"):
                assert abs(float(reference[field].iloc[0]) - getattr(r, field)) < 1e-9, (r, field)
    metrics.to_csv(out / "metrics.csv", index=False)
    diffs.to_csv(out / "paired_differences.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(out / "thresholds.csv", index=False)
    pairs = ablation_inventory(out)
    protocol = [
        {"method": method, "development_predictions_available": method == "P2Rank",
         "status": "development threshold recomputed" if method == "P2Rank" else
                   "final development thresholds available; raw OOF predictions absent" if method == "AdaBind" else
                   "development recalibration cannot be completed from local saved files"}
        for method in ["AdaBind", "bindEmbed21DL", "bindNode24", "ESMBind", "MetalBind", "P2Rank", "AF2BIND"]
    ]
    pd.DataFrame(protocol).to_csv(out / "protocol_inventory.csv", index=False)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True, check=False)
    checkpoint_count = sum(1 for base in (ROOT / "data/drive_inputs", ROOT / "data/_local")
                           for _ in base.rglob("*.pt"))
    metadata = {"bootstrap_replicates": args.boot, "seed": args.seed,
                "prediction_sha256": sha256(PRED), "threshold_manifest_sha256": sha256(MANIFEST),
                "audit_script_sha256": sha256(Path(__file__)),
                "af2bind_scores_sha256": hashlib.sha256(pred.p_af.to_numpy(np.float64).tobytes()).hexdigest(),
                "p2rank_scores_sha256": hashlib.sha256(pred.p_p2.to_numpy(np.float64).tobytes()).hexdigest(),
                "af2bind_reference": "https://doi.org/10.1038/s41592-026-03011-2",
                "af2bind_repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=AF_DIR.parent / "repo", text=True).strip(),
                "rank_metric_intervals": "not recomputed; point values verified invariant to threshold",
                "gpu_command_exit_code": gpu.returncode,
                "gpu_status": gpu.stdout.strip() or gpu.stderr.strip(),
                "local_checkpoint_count_in_cached_inputs": checkpoint_count,
                "training_runs_started": 0, "elapsed_seconds": round(time.monotonic() - started, 2)}
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    af = diffs[(diffs.task == "small") & (diffs.variant == "AF2BIND_published") & diffs.metric.isin(["F1", "MCC"])]
    metal = diffs[(diffs.task == "metal") & (diffs.variant == "ESMBind") &
                  diffs.metric.isin(["F1", "MCC"]) & (diffs.split == "TestSet300")]
    all_fixed = diffs[(diffs.protein_rule == "fixed_all_proteins") &
                      diffs.metric.isin(["F1", "MCC"]) & ~diffs.split.str.endswith("subset66")]
    text = ["# Operating-point audit and ablation readiness", "",
            "This run uses saved predictions only. No model was fitted and no test label selected a threshold.", "",
            "All original main-table point metrics were reproduced before evaluating sensitivity.", "",
            ("Threshold-metric SDs and confidence intervals were also reproduced for the manuscript operating "
             "points before interpreting new comparisons (10,000 replicates, seed 2026)."
             if args.boot == 10000 and args.seed == 2026 else
             "Point-value reproduction gates passed; bootstrap interval reproduction requires 10,000 replicates and seed 2026."), "",
            "## Findings", "",
            "- AdaBind's small-molecule F1/MCC advantage over AF2BIND remains positive at AF2BIND's published 0.28 "
            "threshold, including fixed protein sets, pooled counts and the 66-protein subset.",
            "- Metal F1/MCC comparisons with ESMBind depend on aggregation: the legacy protein means and pooled "
            "counts favor AdaBind, but means on the same protein set favor ESMBind on TestSet300. "
            "Do not describe AdaBind as uniformly superior across threshold metrics.",
            "- P2Rank's development-only thresholds were independently reproduced: 0.4586 for small molecules "
            "and 0.6227 for metal ions. AUPRC/AUROC are invariant to these threshold changes.",
            "- The existing matched seed-0 fold summaries show an ESM-IF1 increase in all five small-molecule folds, "
            "but a slightly negative mean change for merged-label nucleic acids. This is not a new training run.", "",
            "## AF2BIND published-threshold comparison", "",
            "Differences below are AdaBind minus AF2BIND at 0.28, with paired protein-bootstrap 95% intervals.", "",
            markdown_table(af[["split", "protein_rule", "metric", "diff_AdaBind_minus", "ci_lo", "ci_hi"]]),
            "## Metal comparison is sensitive to aggregation", "",
            "On TestSet300 the protein-mean F1/MCC ranking against ESMBind reverses when the same protein set is used. "
            "Pooled-residue metrics and pooled AUPRC address different estimands; report them separately.", "",
            markdown_table(metal[["protein_rule", "metric", "diff_AdaBind_minus", "ci_lo", "ci_hi"]]),
            "## Fixed-all-protein comparisons", "",
            markdown_table(all_fixed[["split", "task", "variant", "metric", "diff_AdaBind_minus", "ci_lo", "ci_hi"]]),
            "## Scope limitations", "",
            "- Step 1 is partial: most baseline development predictions are absent; equal-criterion development recalibration remains pending.",
            "- AF2BIND 0.28 is a published reference threshold, not a development optimum established for the supplied ESMFold structures.",
            "- Rank-metric bootstrap intervals were not rerun because thresholds do not alter rankings.",
            "- Fixed-positive protein means omit false-positive-only proteins; use them alongside fixed-all and original inclusion results.",
            "- On all-negative proteins F1/MCC are zero under the stated convention, even for correct abstention. "
            "Fixed-all and fixed-positive means are sensitivity estimands, not a uniquely correct replacement for the legacy protocol. "
            "Pooled confusion-count metrics retain the effects of false positives across all proteins.",
            "- Step 2 is an inventory and reanalysis of existing fold summaries, not completion of new matched ablations.",
            "- Local checkpoint and embedding caches needed for new ablations are absent; GPU readiness must be restored before a timing pilot.", "",
            "## Existing paired fold summaries", "",
            markdown_table(pd.DataFrame(pairs)[["task", "without", "with", "mean_fold_difference", "sd_fold_difference", "positive_folds"]]), "",
            "## Remaining work", "",
            "1. Restore raw development/OOF predictions for bindEmbed21DL, bindNode24, ESMBind and MetalBind "
            "to compare equally selected development operating points.",
            "2. Generate AF2BIND development predictions if its threshold is to be recalibrated on DevSet1010. "
            "Historical mean inference time is about 21 seconds per test protein; 1,010 such runs would take "
            "about six hours on equivalent hardware, before accounting for differences in sequence lengths.",
            "3. Restore embedding caches/checkpoints and GPU access, run a timing pilot, then execute the missing "
            "matched and random-group ablations. Several hours to days of GPU work should be budgeted; "
            "there is no measured full-training estimate from this local audit.",
            "4. Integrate the sensitivity findings into the manuscript without replacing one aggregation convention "
            "solely because its results favor a particular method."]
    (out / "summary.md").write_text("\n".join(text))
    print("Consistency gates passed. Elapsed seconds:", metadata["elapsed_seconds"], flush=True)
    print(af[["split", "protein_rule", "metric", "diff_AdaBind_minus", "ci_lo", "ci_hi"]].round(4).to_string(index=False))
    print("Reports:", out)


if __name__ == "__main__":
    main()
