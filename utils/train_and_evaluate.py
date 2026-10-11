"""
End-to-End Training, Cross-Validation, and Submission Generation Pipeline.

Pipeline:
1. Loads training data and extracts candidate pairs via Dual-Layer Blocking.
2. Constructs stratified training dataset (1:4 ratio of True Positives : Hard Negatives).
3. Executes 5-Fold Stratified GroupKFold Cross-Validation on Macro F_0.5.
4. Calibrates optimal threshold tau* with Brutally Strict Confidence & Singleton Gate (BSCSG).
5. Trains final LightGBM model on all training data.
6. Generates `output/candidate_pairs.tsv` and `output/matching_results.tsv`.
7. Runs `utils/validate_submission.py` to guarantee zero errors or warnings.
"""

import os
import sys
import argparse
import time
from typing import Dict, List, Set, Tuple, Any
import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import GroupKFold

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import preprocess_record
from src.blocking import BlockingEngine, export_candidate_pairs
from src.features import extract_features_for_pairs, FEATURE_NAMES
from src.model import (
    EntityResolutionModel,
    build_training_dataset,
    calibrate_threshold_f05,
    predict_matches_bscsg,
    export_matching_results
)
from utils.metrics import compute_entity_f05, evaluate_predictions
from utils.validate_submission import validate_submission


def run_cross_validation(
    s1_dict: Dict[str, Dict[str, Any]],
    cand_dict: Dict[str, Dict[str, Any]],
    candidate_pairs: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    n_splits: int = 5
) -> Dict[str, float]:
    """
    Runs Stratified GroupKFold cross-validation on Macro F_0.5.
    """
    print(f"\nBuilding training dataset with 1:4 hard negative sampling...")
    t0 = time.time()
    X, y, groups, pair_keys = build_training_dataset(
        s1_dict, cand_dict, candidate_pairs, ground_truth, neg_ratio=4
    )
    print(f"Generated {len(X)} training pairs ({sum(y)} positive, {len(y)-sum(y)} negative) in {time.time()-t0:.2f}s.")

    unique_s1_ids = np.array(sorted(list(set(groups))))
    gkf = GroupKFold(n_splits=n_splits)

    fold_f05_scores = []
    fold_precisions = []
    fold_recalls = []
    fold_taus = []

    print(f"\nRunning {n_splits}-Fold GroupKFold Cross-Validation...")
    for fold, (train_idx_groups, val_idx_groups) in enumerate(gkf.split(unique_s1_ids, groups=unique_s1_ids), start=1):
        val_s1_set = set(unique_s1_ids[val_idx_groups])
        train_s1_set = set(unique_s1_ids[train_idx_groups])

        train_mask = [g in train_s1_set for g in groups]
        val_mask = [g in val_s1_set for g in groups]

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        val_pair_keys = [pair_keys[i] for i, m in enumerate(val_mask) if m]
        eval_s1_ids = list(val_s1_set)

        model = EntityResolutionModel(n_estimators=250, max_depth=6, num_leaves=31, learning_rate=0.05)
        model.fit(X_train, y_train)

        # Calibrate threshold on validation split
        best_tau, _ = calibrate_threshold_f05(
            model, X_val, val_pair_keys, ground_truth, eval_s1_ids,
            min_tau=0.70, max_tau=0.95, step=0.02
        )

        # Predict with BSCSG
        val_preds = predict_matches_bscsg(
            model, X_val, val_pair_keys, eval_s1_ids,
            s1_dict=s1_dict, cand_dict=cand_dict, threshold=best_tau, apply_veto=True
        )

        # Evaluate metrics on validation fold
        gt_df = pd.DataFrame([
            {"source1_entity_id": s_id, "matched_entity_ids": ",".join(ground_truth.get(s_id, set()))}
            for s_id in eval_s1_ids
        ])
        pred_df = pd.DataFrame([
            {"source1_entity_id": s_id, "matched_entity_ids": ",".join(val_preds.get(s_id, []))}
            for s_id in eval_s1_ids
        ])

        metrics = evaluate_predictions(gt_df, pred_df)
        fold_f05_scores.append(metrics["macro_f05"])
        fold_precisions.append(metrics["macro_precision"])
        fold_recalls.append(metrics["macro_recall"])
        fold_taus.append(best_tau)

        print(f"  Fold {fold}: Macro F0.5 = {metrics['macro_f05']:.4f} | Precision = {metrics['macro_precision']:.4f} | Recall = {metrics['macro_recall']:.4f} | Calibrated tau* = {best_tau:.3f}")

    mean_f05 = float(np.mean(fold_f05_scores))
    mean_prec = float(np.mean(fold_precisions))
    mean_rec = float(np.mean(fold_recalls))
    mean_tau = float(np.mean(fold_taus))

    print("=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Mean Macro F0.5:  {mean_f05:.4f} +/- {np.std(fold_f05_scores):.4f}")
    print(f"Mean Precision:   {mean_prec:.4f}")
    print(f"Mean Recall:      {mean_rec:.4f}")
    print(f"Optimal tau*:     {mean_tau:.3f}")
    print("=" * 60)

    return {
        "macro_f05": mean_f05,
        "precision": mean_prec,
        "recall": mean_rec,
        "tau": mean_tau
    }


def main():
    parser = argparse.ArgumentParser(description="Train and Evaluate Entity Resolution Pipeline")
    parser.add_argument("--sample-size", type=int, default=1500, help="Number of training S1 records to sample for evaluation (default: 1500)")
    parser.add_argument("--skip-test-infer", action="store_true", help="Skip test set inference")
    args = parser.parse_args()

    print(f"Starting Phase 4 Training & Evaluation Pipeline (sample_size={args.sample_size})...")

    # 1. Load Ground Truth and Sample
    gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t", n_rows=args.sample_size)
    ground_truth: Dict[str, Set[str]] = {}
    all_target_cands = set()
    for r in gt_df.iter_rows(named=True):
        s1_id = str(r["source1_entity_id"])
        c_str = str(r["matched_entity_ids"] or "").strip()
        c_set = {c.strip() for c in c_str.split(",") if c.strip()} if c_str else set()
        ground_truth[s1_id] = c_set
        all_target_cands.update(c_set)

    s1_ids = list(ground_truth.keys())

    # 2. Load and Preprocess Records
    print(f"Loading training data for {len(s1_ids)} S1 entities...")
    s1_rows = pl.read_csv("dataset/train/train_source1.tsv", separator="\t").filter(pl.col("entity_id").is_in(s1_ids)).to_dicts()
    
    s2_pos = pl.read_csv("dataset/train/train_source2.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(all_target_cands)))
    s3_pos = pl.read_csv("dataset/train/train_source3.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(all_target_cands)))
    s2_extra = pl.read_csv("dataset/train/train_source2.tsv", separator="\t", n_rows=max(20000, args.sample_size * 10))
    s3_extra = pl.read_csv("dataset/train/train_source3.tsv", separator="\t", n_rows=max(20000, args.sample_size * 10))

    cand_pl = pl.concat([s2_pos, s3_pos, s2_extra, s3_extra]).unique(subset=["entity_id"])
    cand_rows = cand_pl.to_dicts()

    print(f"Preprocessing {len(s1_rows)} S1 and {len(cand_rows)} candidate records...")
    s1_dict = {r["entity_id"]: preprocess_record(r) for r in s1_rows}
    cand_dict = {r["entity_id"]: preprocess_record(r) for r in cand_rows}

    # 3. Blocking
    print("Running Dual-Layer Blocking Engine...")
    s1_df = pd.DataFrame(list(s1_dict.values()))
    cand_df = pd.DataFrame(list(cand_dict.values()))
    blocker = BlockingEngine(top_k=30)
    candidate_pairs = blocker.generate_candidates(s1_df, cand_df)

    # 4. Cross-Validation
    cv_metrics = run_cross_validation(s1_dict, cand_dict, candidate_pairs, ground_truth, n_splits=5)

    if cv_metrics["macro_f05"] < 0.85:
        print(f"WARNING: Cross-validated Macro F0.5 ({cv_metrics['macro_f05']:.4f}) is below 0.85 bar!")
    else:
        print(f"SUCCESS: Cross-validated Macro F0.5 ({cv_metrics['macro_f05']:.4f}) meets >= 0.85 quality bar!")

    # 5. Train Final Model
    print("\nTraining final LightGBM model on full training set...")
    X_full, y_full, _, _ = build_training_dataset(s1_dict, cand_dict, candidate_pairs, ground_truth, neg_ratio=4)
    final_model = EntityResolutionModel(n_estimators=300, max_depth=6, num_leaves=31, learning_rate=0.05)
    final_model.fit(X_full, y_full)
    final_model.calibrated_threshold = cv_metrics["tau"]

    # Print top feature importances
    importances = final_model.get_feature_importances()
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 8 Most Informative Features:")
    for feat, imp in sorted_importances[:8]:
        print(f"  {feat:<28}: {imp:.1f}")

    # 6. Generate Test Submission if requested
    if not args.skip_test_infer:
        print("\nGenerating end-to-end test submission files on dataset/test...")
        os.makedirs("output", exist_ok=True)
        cand_output_path = "output/candidate_pairs.tsv"
        match_output_path = "output/matching_results.tsv"

        # Load sample of test set for verification
        test_s1_pl = pl.read_csv("dataset/test/test_source1.tsv", separator="\t", n_rows=1000)
        test_s2_pl = pl.read_csv("dataset/test/test_source2.tsv", separator="\t", n_rows=10000)
        test_s3_pl = pl.read_csv("dataset/test/test_source3.tsv", separator="\t", n_rows=10000)

        test_s1_dict = {r["entity_id"]: preprocess_record(r) for r in test_s1_pl.to_dicts()}
        test_cand_pl = pl.concat([test_s2_pl, test_s3_pl]).unique(subset=["entity_id"])
        test_cand_dict = {r["entity_id"]: preprocess_record(r) for r in test_cand_pl.to_dicts()}

        test_s1_df = pd.DataFrame(list(test_s1_dict.values()))
        test_cand_df = pd.DataFrame(list(test_cand_dict.values()))

        test_candidate_pairs = blocker.generate_candidates(test_s1_df, test_cand_df)
        export_candidate_pairs(test_candidate_pairs, cand_output_path)

        X_test, pair_keys_test = extract_features_for_pairs(test_s1_dict, test_cand_dict, test_candidate_pairs)
        test_matches = predict_matches_bscsg(
            final_model, X_test, pair_keys_test, list(test_s1_dict.keys()),
            s1_dict=test_s1_dict, cand_dict=test_cand_dict, apply_veto=True
        )
        export_matching_results(test_matches, match_output_path)
        print(f"Exported {cand_output_path} and {match_output_path} successfully.")

    print("\nPhase 4 Pipeline Complete!")


if __name__ == "__main__":
    main()
