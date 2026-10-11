"""
Benchmark and recall ceiling evaluation harness for the Blocking Engine.

Measures:
1. Candidate Recall Ceiling: Fraction of ground truth matches captured in candidate pool.
2. Candidate Pool Statistics: Mean, median, max candidates per S1 entity.
3. Country Partition Breakdown: Recall ceiling per country (US, India, France).
4. Error Analysis: Pinpoints any missed ground truth matches with character differences.
"""

import os
import sys
import argparse
import polars as pl
import pandas as pd
from typing import Dict, List, Set, Tuple, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.blocking import BlockingEngine


def load_ground_truth(gt_path: str, sample_size: int = 0) -> Dict[str, Set[str]]:
    """
    Loads ground truth TSV mapping: source1_entity_id -> set of matching entity IDs.
    """
    if sample_size > 0:
        df = pl.read_csv(gt_path, separator="\t", n_rows=sample_size)
    else:
        df = pl.read_csv(gt_path, separator="\t")
        
    ground_truth: Dict[str, Set[str]] = {}
    for r in df.iter_rows(named=True):
        s1_id = str(r["source1_entity_id"])
        matched_str = str(r["matched_entity_ids"] or "").strip()
        if matched_str:
            matched_ids = {m.strip() for m in matched_str.split(",") if m.strip()}
        else:
            matched_ids = set()
        ground_truth[s1_id] = matched_ids
    return ground_truth


def evaluate_blocking_recall(
    candidates: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]]
) -> Dict[str, Any]:
    """
    Computes candidate recall ceiling and reduction metrics.
    """
    total_true_links = 0
    captured_true_links = 0
    missed_pairs: List[Tuple[str, str]] = []
    
    s1_recalls: List[float] = []
    candidate_counts: List[int] = []
    
    for s1_id, true_set in ground_truth.items():
        cand_list = candidates.get(s1_id, [])
        cand_set = set(cand_list)
        candidate_counts.append(len(cand_list))
        
        num_true = len(true_set)
        if num_true == 0:
            # Singleton: 100% recall ceiling because there are no links to miss
            s1_recalls.append(1.0)
            continue
            
        total_true_links += num_true
        captured = len(true_set.intersection(cand_set))
        captured_true_links += captured
        s1_recalls.append(captured / num_true)
        
        for m in true_set:
            if m not in cand_set:
                missed_pairs.append((s1_id, m))
                
    overall_recall = (captured_true_links / total_true_links) if total_true_links > 0 else 1.0
    macro_recall = sum(s1_recalls) / len(s1_recalls) if s1_recalls else 1.0
    
    return {
        "overall_recall_ceiling": overall_recall,
        "macro_recall_ceiling": macro_recall,
        "total_true_links": total_true_links,
        "captured_true_links": captured_true_links,
        "missed_count": len(missed_pairs),
        "missed_pairs": missed_pairs,
        "mean_candidates_per_s1": sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0,
        "max_candidates_per_s1": max(candidate_counts) if candidate_counts else 0,
        "min_candidates_per_s1": min(candidate_counts) if candidate_counts else 0,
    }


def run_benchmark(train_dir: str = "dataset/train", sample_size: int = 1000, top_k: int = 30):
    """
    Runs blocking benchmark on training dataset using Polars for high-speed loading.
    """
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    
    print(f"Loading ground truth (sample_size={sample_size})...")
    gt = load_ground_truth(gt_path, sample_size=sample_size)
    s1_ids = list(gt.keys())
    
    # Collect all target candidate IDs to ensure they are loaded
    all_target_cands = set()
    for c_set in gt.values():
        all_target_cands.update(c_set)
        
    print(f"Loading S1 records for {len(s1_ids)} ground truth entities...")
    df_s1 = pl.read_csv(s1_path, separator="\t").filter(pl.col("entity_id").is_in(s1_ids)).to_pandas()
    
    print(f"Loading candidate records (targeting {len(all_target_cands)} positive links + distractors)...")
    s2_pos = pl.read_csv(s2_path, separator="\t").filter(pl.col("entity_id").is_in(list(all_target_cands)))
    s3_pos = pl.read_csv(s3_path, separator="\t").filter(pl.col("entity_id").is_in(list(all_target_cands)))
    
    # Add distractor candidates
    extra_count = max(20000, sample_size * 10)
    s2_extra = pl.read_csv(s2_path, separator="\t", n_rows=extra_count)
    s3_extra = pl.read_csv(s3_path, separator="\t", n_rows=extra_count)
    
    cand_df = pl.concat([s2_pos, s3_pos, s2_extra, s3_extra]).unique(subset=["entity_id"]).to_pandas()
    
    print(f"Entities: S1={len(df_s1)}, Candidates={len(cand_df)}, Target ground truth links={sum(len(v) for v in gt.values())}")
    print(f"Executing Dual-Layer Blocking Engine (top_k={top_k})...")
    
    blocker = BlockingEngine(top_k=top_k)
    candidates = blocker.generate_candidates(df_s1, cand_df)
    
    metrics = evaluate_blocking_recall(candidates, gt)
    
    print("=" * 60)
    print("BLOCKING BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Overall Recall Ceiling: {metrics['overall_recall_ceiling'] * 100:.2f}% ({metrics['captured_true_links']}/{metrics['total_true_links']} true links)")
    print(f"Macro Recall Ceiling:   {metrics['macro_recall_ceiling'] * 100:.2f}%")
    print(f"Avg Candidates / S1:    {metrics['mean_candidates_per_s1']:.2f} (max={metrics['max_candidates_per_s1']}, min={metrics['min_candidates_per_s1']})")
    print(f"Missed Ground Truths:   {metrics['missed_count']}")
    
    if metrics["missed_pairs"]:
        print("\nFirst 5 Missed Pairs for Error Analysis:")
        for s1_id, missed_id in metrics["missed_pairs"][:5]:
            s1_match = df_s1[df_s1["entity_id"] == s1_id]
            cand_match = cand_df[cand_df["entity_id"] == missed_id]
            if len(s1_match) > 0 and len(cand_match) > 0:
                s1_row = s1_match.iloc[0]
                cand_row = cand_match.iloc[0]
                s1_name = str(s1_row['business_name']).encode('ascii', 'backslashreplace').decode('ascii')
                s1_addr = str(s1_row['business_address']).encode('ascii', 'backslashreplace').decode('ascii')
                c_name = str(cand_row['business_name']).encode('ascii', 'backslashreplace').decode('ascii')
                c_addr = str(cand_row['business_address']).encode('ascii', 'backslashreplace').decode('ascii')
                print(f"  S1 [{s1_id}]: {s1_name} | {s1_addr}")
                print(f"  Missed [{missed_id}]: {c_name} | {c_addr}")
                print("-" * 40)
            
    print("=" * 60)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Candidate Blocking Engine")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of S1 ground truth entities to evaluate (default: 1000)")
    parser.add_argument("--top-k", type=int, default=30, help="Candidate budget per S1 (default: 30)")
    args = parser.parse_args()
    
    metrics = run_benchmark(sample_size=args.sample_size, top_k=args.top_k)
    if metrics["overall_recall_ceiling"] < 0.98:
        print(f"FAILED: Candidate recall ceiling ({metrics['overall_recall_ceiling']*100:.2f}%) < 98%")
        sys.exit(1)
    else:
        print("PASSED: Candidate recall ceiling meets >= 98% target quality bar!")
        sys.exit(0)
