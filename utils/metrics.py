"""
Evaluation metrics for the Business Entity Resolution Challenge.

Evaluates predictions using Macro-averaged F_beta Score (beta = 0.5):
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

Special Rules from Problem Statement:
- Macro-averaged per Source 1 entity across all Source 1 entities in the evaluation set.
- Singletons (Source 1 entity with no true matches):
    - Scores 1.0 when correctly predicted as empty.
    - Scores 0.0 when any match is predicted.
- If true matches > 0 and predicted matches == 0:
    - Precision = 0.0, Recall = 0.0 -> F_0.5 = 0.0.
"""

from typing import Dict, List, Set, Union, Tuple
import pandas as pd


def compute_entity_f05(true_matches: Set[str], pred_matches: Set[str]) -> Tuple[float, float, float]:
    """
    Computes precision, recall, and F_0.5 score for a single Source 1 entity.
    
    Args:
        true_matches: Set of ground-truth matching entity IDs (S2-/S3-)
        pred_matches: Set of predicted matching entity IDs (S2-/S3-)
        
    Returns:
        (precision, recall, f05)
    """
    num_true = len(true_matches)
    num_pred = len(pred_matches)
    
    # Singleton case: entity has no true matches
    if num_true == 0:
        if num_pred == 0:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0
            
    # Non-singleton but no predictions made
    if num_pred == 0:
        return 0.0, 0.0, 0.0
        
    true_positives = len(true_matches.intersection(pred_matches))
    if true_positives == 0:
        return 0.0, 0.0, 0.0
        
    precision = true_positives / num_pred
    recall = true_positives / num_true
    
    # F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    denom = 0.25 * precision + recall
    if denom == 0.0:
        f05 = 0.0
    else:
        f05 = (1.25 * precision * recall) / denom
        
    return precision, recall, f05


def parse_id_list(val: Union[str, float, None]) -> Set[str]:
    """Parses a comma-separated entity ID string into a set of unique IDs."""
    if val is None or pd.isna(val):
        return set()
    val_str = str(val).strip()
    if not val_str:
        return set()
    return {item.strip() for item in val_str.split(",") if item.strip()}


def evaluate_predictions(
    ground_truth_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    s1_col_gt: str = "source1_entity_id",
    matches_col_gt: str = "matched_entity_ids",
    s1_col_pred: str = "source1_entity_id",
    matches_col_pred: str = "matched_entity_ids"
) -> Dict[str, float]:
    """
    Evaluates predictions against ground truth and returns macro-averaged metrics.
    
    Args:
        ground_truth_df: DataFrame with ground truth source1_entity_id and matched_entity_ids
        predictions_df: DataFrame with predicted source1_entity_id and matched_entity_ids
        
    Returns:
        Dictionary containing:
            - 'macro_f05': Macro-averaged F_0.5 score
            - 'macro_precision': Macro-averaged Precision
            - 'macro_recall': Macro-averaged Recall
            - 'singleton_accuracy': Accuracy on singleton entities
            - 'num_entities': Total number of Source 1 entities evaluated
            - 'num_singletons': Number of true singletons
    """
    # Parse ground truth
    gt_map: Dict[str, Set[str]] = {}
    for _, row in ground_truth_df.iterrows():
        s1_id = str(row[s1_col_gt]).strip()
        gt_map[s1_id] = parse_id_list(row[matches_col_gt])
        
    # Parse predictions
    pred_map: Dict[str, Set[str]] = {}
    for _, row in predictions_df.iterrows():
        s1_id = str(row[s1_col_pred]).strip()
        pred_map[s1_id] = parse_id_list(row[matches_col_pred])
        
    f05_scores: List[float] = []
    precision_scores: List[float] = []
    recall_scores: List[float] = []
    
    singleton_total = 0
    singleton_correct = 0
    
    for s1_id, true_set in gt_map.items():
        pred_set = pred_map.get(s1_id, set())
        
        prec, rec, f05 = compute_entity_f05(true_set, pred_set)
        f05_scores.append(f05)
        precision_scores.append(prec)
        recall_scores.append(rec)
        
        if len(true_set) == 0:
            singleton_total += 1
            if len(pred_set) == 0:
                singleton_correct += 1
                
    num_entities = len(gt_map)
    macro_f05 = sum(f05_scores) / num_entities if num_entities > 0 else 0.0
    macro_prec = sum(precision_scores) / num_entities if num_entities > 0 else 0.0
    macro_rec = sum(recall_scores) / num_entities if num_entities > 0 else 0.0
    singleton_acc = singleton_correct / singleton_total if singleton_total > 0 else 1.0
    
    return {
        "macro_f05": macro_f05,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "singleton_accuracy": singleton_acc,
        "num_entities": num_entities,
        "num_singletons": singleton_total
    }
