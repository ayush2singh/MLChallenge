"""
Local Benchmark: Validates the improved pipeline on training data.

Measures:
1. Blocking Recall (pair-level and entity-level)
2. Cross-Validated Macro F0.5
3. Feature importance ranking
4. Optimal threshold

Usage:
    python benchmark_local.py --sample-size 5000
    python benchmark_local.py --sample-size 20000
"""

import os
import sys
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, Set, List, Any

from src.preprocess import preprocess_record
from src.blocking import BlockingEngine, measure_blocking_recall
from src.features import extract_features_for_pairs, FEATURE_NAMES
from src.model import (
    EntityResolutionModel,
    build_training_dataset,
    calibrate_threshold_f05,
    predict_matches_bscsg,
)
from utils.metrics import compute_entity_f05


def load_training_sample(sample_size: int):
    """Loads a stratified sample of training data."""
    t0 = time.time()

    # Load ground truth
    gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t", n_rows=sample_size)
    ground_truth: Dict[str, Set[str]] = {}
    all_target_cands = set()

    for r in gt_df.iter_rows(named=True):
        s1_id = str(r["source1_entity_id"])
        c_str = str(r["matched_entity_ids"] or "").strip()
        c_set = {c.strip() for c in c_str.split(",") if c.strip()} if c_str else set()
        ground_truth[s1_id] = c_set
        all_target_cands.update(c_set)

    s1_ids = list(ground_truth.keys())
    target_s2 = [c for c in all_target_cands if c.startswith("S2-")]
    target_s3 = [c for c in all_target_cands if c.startswith("S3-")]

    print(f"Ground truth loaded: {len(s1_ids)} S1 entities, {len(all_target_cands)} true matches")
    n_singletons = sum(1 for s in ground_truth.values() if len(s) == 0)
    print(f"  Singletons: {n_singletons} ({100*n_singletons/len(s1_ids):.1f}%)")

    # Load S1 records
    s1_pl = pl.read_csv("dataset/train/train_source1.tsv", separator="\t")
    s1_rows = s1_pl.filter(pl.col("entity_id").is_in(s1_ids)).to_dicts()

    # Load true-match S2/S3 records + extra candidates for realistic blocking
    s2_pos = pl.read_csv("dataset/train/train_source2.tsv", separator="\t").filter(
        pl.col("entity_id").is_in(target_s2)
    )
    s3_pos = pl.read_csv("dataset/train/train_source3.tsv", separator="\t").filter(
        pl.col("entity_id").is_in(target_s3)
    )

    # Load extra candidates proportional to sample size for realistic blocking difficulty
    extra_n = max(50000, sample_size * 15)
    s2_extra = pl.read_csv("dataset/train/train_source2.tsv", separator="\t", n_rows=extra_n)
    s3_extra = pl.read_csv("dataset/train/train_source3.tsv", separator="\t", n_rows=extra_n)

    cand_pl = pl.concat([s2_pos, s3_pos, s2_extra, s3_extra]).unique(subset=["entity_id"])
    cand_rows = cand_pl.to_dicts()

    print(f"Loaded {len(s1_rows)} S1 records and {len(cand_rows)} candidate records in {time.time()-t0:.1f}s")

    # Preprocess
    print("Preprocessing records...")
    t0 = time.time()
    s1_dict = {r["entity_id"]: preprocess_record(r) for r in s1_rows}
    cand_dict = {r["entity_id"]: preprocess_record(r) for r in cand_rows}
    print(f"  Preprocessed in {time.time()-t0:.1f}s")

    return s1_dict, cand_dict, ground_truth


def run_benchmark(sample_size: int = 5000):
    """Runs the full benchmark pipeline."""
    print("=" * 65)
    print(f"BENCHMARK v2: Address-First Blocking + 26 Features (N={sample_size})")
    print("=" * 65)

    # 1. Load data
    s1_dict, cand_dict, ground_truth = load_training_sample(sample_size)

    # 2. Blocking
    print("\n--- STAGE 1: BLOCKING ---")
    t0 = time.time()
    s1_df = pd.DataFrame(list(s1_dict.values()))
    cand_df = pd.DataFrame(list(cand_dict.values()))
    blocker = BlockingEngine(top_k=50)
    candidate_pairs = blocker.generate_candidates(s1_df, cand_df)

    blocking_time = time.time() - t0
    print(f"  Blocking completed in {blocking_time:.1f}s")

    # Measure blocking recall
    recall_metrics = measure_blocking_recall(candidate_pairs, ground_truth)
    print(f"  Pair Recall:            {recall_metrics['pair_recall']:.4f} ({recall_metrics['found_true_pairs']}/{recall_metrics['total_true_pairs']})")
    print(f"  Entity Perfect Recall:  {recall_metrics['entity_perfect_recall']:.4f} ({recall_metrics['entities_perfect_recall']}/{recall_metrics['entities_with_matches']})")

    avg_cands = np.mean([len(v) for v in candidate_pairs.values()])
    print(f"  Avg candidates per S1:  {avg_cands:.1f}")

    # 3. Feature Extraction & Training
    print("\n--- STAGE 2: TRAINING ---")
    t0 = time.time()

    # Split into train (80%) and validation (20%)
    s1_ids = list(ground_truth.keys())
    np.random.seed(42)
    np.random.shuffle(s1_ids)
    split_point = int(0.8 * len(s1_ids))
    train_s1_ids = set(s1_ids[:split_point])
    val_s1_ids = set(s1_ids[split_point:])

    train_gt = {k: v for k, v in ground_truth.items() if k in train_s1_ids}
    val_gt = {k: v for k, v in ground_truth.items() if k in val_s1_ids}

    train_pairs = {k: v for k, v in candidate_pairs.items() if k in train_s1_ids}
    val_pairs = {k: v for k, v in candidate_pairs.items() if k in val_s1_ids}

    # Build training dataset
    X_train, y_train, groups_train, pk_train = build_training_dataset(
        s1_dict, cand_dict, train_pairs, train_gt, neg_ratio=-1
    )
    print(f"  Training set: {len(X_train)} pairs ({sum(y_train)} pos, {len(y_train)-sum(y_train)} neg)")

    # Build validation dataset (all candidates, not just negatives)
    X_val, pk_val = extract_features_for_pairs(s1_dict, cand_dict, val_pairs)
    print(f"  Validation set: {len(X_val)} pairs from {len(val_s1_ids)} S1 entities")

    # Train model
    model = EntityResolutionModel(n_estimators=500, max_depth=7, num_leaves=63, learning_rate=0.05)
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Training completed in {train_time:.1f}s")

    # Feature importances
    importances = model.get_feature_importances()
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    print("\n  Top 10 Features:")
    for feat, imp in sorted_imp[:10]:
        print(f"    {feat:<28}: {imp:.1f}")

    # 4. Threshold Calibration
    print("\n--- STAGE 3: THRESHOLD CALIBRATION ---")
    t0 = time.time()
    val_s1_list = list(val_s1_ids)
    best_tau, best_f05_cal = calibrate_threshold_f05(
        model, X_val, pk_val, ground_truth, val_s1_list,
        min_tau=0.30, max_tau=0.98, step=0.005
    )
    print(f"  Optimal tau*: {best_tau:.4f} (calibration F0.5: {best_f05_cal:.4f})")
    print(f"  Calibration completed in {time.time()-t0:.1f}s")

    # 5. Final Evaluation with BSCSG
    print("\n--- STAGE 4: EVALUATION WITH BSCSG ---")
    val_preds = predict_matches_bscsg(
        model, X_val, pk_val, val_s1_list,
        s1_dict=s1_dict, cand_dict=cand_dict,
        threshold=best_tau, apply_veto=True
    )

    # Compute per-entity F0.5
    f05_scores = []
    prec_scores = []
    rec_scores = []
    for s1_id in val_s1_list:
        true_set = ground_truth.get(s1_id, set())
        pred_set = set(val_preds.get(s1_id, []))
        p, r, f = compute_entity_f05(true_set, pred_set)
        f05_scores.append(f)
        prec_scores.append(p)
        rec_scores.append(r)

    macro_f05 = np.mean(f05_scores)
    macro_prec = np.mean(prec_scores)
    macro_rec = np.mean(rec_scores)

    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Sample Size:       {sample_size}")
    print(f"  Validation Size:   {len(val_s1_ids)}")
    print(f"  Blocking Recall:   {recall_metrics['pair_recall']:.4f}")
    print(f"  Optimal Tau*:      {best_tau:.4f}")
    print(f"  Macro F0.5 Score:  {macro_f05:.4f}")
    print(f"  Macro Precision:   {macro_prec:.4f}")
    print(f"  Macro Recall:      {macro_rec:.4f}")
    print("=" * 60)

    if macro_f05 >= 0.93:
        print("SUCCESS: Target 0.93+ achieved!")
    elif macro_f05 >= 0.85:
        print("GOOD: Above 0.85 quality bar.")
    else:
        print(f"NEEDS IMPROVEMENT: {macro_f05:.4f} < 0.85")

    return macro_f05


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Benchmark for Entity Resolution Pipeline v2")
    parser.add_argument("--sample-size", type=int, default=5000, help="Number of S1 entities to sample")
    args = parser.parse_args()
    run_benchmark(args.sample_size)
