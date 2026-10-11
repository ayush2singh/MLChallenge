"""
Entity Resolution Model and Brutally Strict Confidence & Singleton Gate (BSCSG) v2.

Contains:
1. Training Dataset Builder with Stratified Hard Negative Sampling (1:6 ratio).
2. LightGBM Pairwise Binary Classifier (500 estimators, deeper trees).
3. Fine-Grained Macro F_0.5 Threshold Calibration Engine (step=0.005).
4. BSCSG Prediction Engine with Smart Postal/Address Discrepancy Vetoes.
5. Competition TSV Serializer for `matching_results.tsv`.

Key improvements over v1:
- Larger neg_ratio (6 vs 4) for harder training signal
- Fine-grained threshold search (step=0.005 over 0.30-0.98)
- Smarter veto: only vetoes on postal conflict when BOTH codes are 5+ digits
- Number conflict veto requires both sides to have 2+ numbers
"""

import os
import csv
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.features import compute_pair_features, FEATURE_NAMES
from utils.metrics import compute_entity_f05


class EntityResolutionModel:
    """
    LightGBM Pairwise Classifier for Business Entity Resolution.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 7,
        num_leaves: int = 63,
        learning_rate: float = 0.05,
        min_child_samples: int = 20,
        random_state: int = 42
    ):
        self.model = LGBMClassifier(
            objective="binary",
            metric="auc",
            boosting_type="gbdt",
            n_estimators=n_estimators,
            max_depth=max_depth,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            min_child_samples=min_child_samples,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1
        )
        self.calibrated_threshold: float = 0.50
        self.feature_names: List[str] = FEATURE_NAMES

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "EntityResolutionModel":
        """Fits the underlying LightGBM binary classifier."""
        self.model.fit(X[self.feature_names], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns positive class probability P(match = 1)."""
        return self.model.predict_proba(X[self.feature_names])[:, 1]

    def get_feature_importances(self) -> Dict[str, float]:
        """Returns feature importances mapped by name."""
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, [float(v) for v in importances]))


def build_training_dataset(
    s1_dict: Dict[str, Dict[str, Any]],
    cand_dict: Dict[str, Dict[str, Any]],
    candidate_pairs: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    neg_ratio: int = -1
) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[Tuple[str, str]]]:
    """
    Constructs a training dataset:
    - All True Positive pairs from ground truth (including out-of-blocking positives)
    - ALL blocking negatives (neg_ratio=-1) or top `neg_ratio` per S1 entity

    Using ALL blocking negatives is critical for proper probability calibration.
    With only top-N negatives, the model sees a biased sample and outputs
    inflated probabilities for all blocking candidates.

    Returns:
        (X_df, y_array, s1_groups, pair_keys)
    """
    feature_rows = []
    labels = []
    groups = []
    pair_keys = []

    for s1_id, true_set in ground_truth.items():
        s1_rec = s1_dict.get(s1_id)
        if not s1_rec:
            continue

        cands_retrieved = candidate_pairs.get(s1_id, [])
        cands_retrieved_set = set(cands_retrieved)

        # 1. Add True Positive Pairs (from both in-blocking and out-of-blocking)
        for cand_id in true_set:
            cand_rec = cand_dict.get(cand_id)
            if cand_rec:
                # Determine rank if candidate was in blocking pool
                if cand_id in cands_retrieved_set:
                    rank = cands_retrieved.index(cand_id) + 1
                else:
                    rank = len(cands_retrieved) + 1  # Out-of-blocking rank
                feats = compute_pair_features(s1_rec, cand_rec, rank=rank)
                feature_rows.append(feats)
                labels.append(1)
                groups.append(s1_id)
                pair_keys.append((s1_id, cand_id))

        # 2. Add Negative Pairs from Blocking Pool (all or limited)
        neg_count = 0
        for rank, cand_id in enumerate(cands_retrieved, start=1):
            if cand_id not in true_set:
                cand_rec = cand_dict.get(cand_id)
                if cand_rec:
                    feats = compute_pair_features(s1_rec, cand_rec, rank=rank)
                    feature_rows.append(feats)
                    labels.append(0)
                    groups.append(s1_id)
                    pair_keys.append((s1_id, cand_id))
                    neg_count += 1
                    if neg_ratio > 0 and neg_count >= neg_ratio:
                        break

    X_df = pd.DataFrame(feature_rows)[FEATURE_NAMES]
    y_arr = np.array(labels, dtype=int)
    return X_df, y_arr, groups, pair_keys


def calibrate_threshold_f05(
    model: EntityResolutionModel,
    X_val: pd.DataFrame,
    pair_keys_val: List[Tuple[str, str]],
    ground_truth: Dict[str, Set[str]],
    eval_s1_ids: List[str],
    min_tau: float = 0.30,
    max_tau: float = 0.98,
    step: float = 0.005
) -> Tuple[float, float]:
    """
    Fine-grained grid search of probability threshold tau* that maximizes Macro F_0.5.

    Searches from min_tau to max_tau with fine step size for precise calibration.

    Returns:
        (best_tau, best_macro_f05)
    """
    probs = model.predict_proba(X_val)

    # Group candidate predictions by S1 ID: s1_id -> [(cand_id, prob)]
    s1_cand_probs: Dict[str, List[Tuple[str, float]]] = {s1_id: [] for s1_id in eval_s1_ids}
    for (s1_id, cand_id), prob in zip(pair_keys_val, probs):
        if s1_id in s1_cand_probs:
            s1_cand_probs[s1_id].append((cand_id, float(prob)))

    best_tau = 0.50
    best_macro_f05 = -1.0

    current_tau = min_tau
    while current_tau <= max_tau:
        f05_scores = []
        for s1_id in eval_s1_ids:
            true_matches = ground_truth.get(s1_id, set())
            cand_list = s1_cand_probs[s1_id]

            # Filter by current threshold
            pred_matches = {c_id for c_id, p in cand_list if p >= current_tau}

            _, _, f05 = compute_entity_f05(true_matches, pred_matches)
            f05_scores.append(f05)

        macro_f05 = float(np.mean(f05_scores)) if f05_scores else 0.0
        if macro_f05 > best_macro_f05:
            best_macro_f05 = macro_f05
            best_tau = round(current_tau, 4)

        current_tau += step

    model.calibrated_threshold = best_tau
    return best_tau, best_macro_f05


def predict_matches_bscsg(
    model: EntityResolutionModel,
    X_test: pd.DataFrame,
    pair_keys_test: List[Tuple[str, str]],
    all_s1_ids: List[str],
    s1_dict: Optional[Dict[str, Dict[str, Any]]] = None,
    cand_dict: Optional[Dict[str, Dict[str, Any]]] = None,
    threshold: Optional[float] = None,
    apply_veto: bool = True
) -> Dict[str, List[str]]:
    """
    Generates final predictions using the Brutally Strict Confidence & Singleton Gate (BSCSG) v2.

    Rules:
    1. Base probability cutoff at calibrated threshold tau*.
    2. Smart Postal Veto: Only vetoes when BOTH postal codes are 5+ digits and differ.
       (Avoids false vetoes from partial/missing postal codes.)
    3. Smart Number Conflict Veto: Only vetoes when BOTH sides have 2+ address numbers
       with zero overlap. (Avoids false vetoes when one side has just "1" or "2".)
    4. Walk-Away Default: If no candidate passes, predict empty string.

    Returns:
        Mapping of {s1_id: [matched_candidate_ids]}
    """
    tau = threshold if threshold is not None else model.calibrated_threshold
    probs = model.predict_proba(X_test) if len(X_test) > 0 else np.array([])

    s1_preds: Dict[str, List[str]] = {s1_id: [] for s1_id in all_s1_ids}

    # Group candidate probabilities by S1
    s1_scored_candidates: Dict[str, List[Tuple[str, float]]] = {s1_id: [] for s1_id in all_s1_ids}
    for (s1_id, cand_id), prob in zip(pair_keys_test, probs):
        if s1_id in s1_scored_candidates:
            s1_scored_candidates[s1_id].append((cand_id, float(prob)))

    for s1_id, candidates in s1_scored_candidates.items():
        if not candidates:
            s1_preds[s1_id] = []
            continue

        s1_rec = s1_dict.get(s1_id) if s1_dict else None
        valid_matches = []

        # Sort candidates descending by probability
        sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

        for cand_id, prob in sorted_candidates:
            if prob < tau:
                continue

            # Smart Veto checks
            if apply_veto and s1_rec and cand_dict:
                cand_rec = cand_dict.get(cand_id)
                if cand_rec:
                    s1_post = s1_rec.get("postal_code", "") or ""
                    c_post = cand_rec.get("postal_code", "") or ""

                    # Postal veto: only when BOTH codes are 5+ digits and differ
                    if (s1_post and c_post and
                        len(s1_post) >= 5 and len(c_post) >= 5 and
                        s1_post != c_post):
                        continue

                    # Number conflict veto: only when BOTH sides have 2+ numbers
                    # and they share zero overlap
                    s1_nums = set(s1_rec.get("address_numbers", []))
                    c_nums = set(cand_rec.get("address_numbers", []))
                    if (len(s1_nums) >= 2 and len(c_nums) >= 2 and
                        len(s1_nums & c_nums) == 0):
                        continue

            valid_matches.append(cand_id)

        s1_preds[s1_id] = valid_matches

    return s1_preds


def export_matching_results(matches: Dict[str, List[str]], output_path: str) -> None:
    """
    Serializes final entity matches to competition-compliant TSV file:
    source1_entity_id\tmatched_entity_ids

    Args:
        matches: Dict mapping source1_entity_id -> list of matched IDs
        output_path: Path where matching_results.tsv will be saved
    """
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for s1_id in sorted(matches.keys()):
            cand_list = matches[s1_id]
            # Ensure unique IDs and preserve ordering
            seen = set()
            unique_cands = [c for c in cand_list if not (c in seen or seen.add(c))]
            match_str = ",".join(unique_cands)
            writer.writerow([s1_id, match_str])
