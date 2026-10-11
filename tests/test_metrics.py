"""
Unit tests for utils/metrics.py
"""

import pytest
import pandas as pd
from utils.metrics import compute_entity_f05, evaluate_predictions


def test_singleton_correct_empty():
    """Singleton correctly predicted as empty should score 1.0."""
    p, r, f = compute_entity_f05(set(), set())
    assert p == 1.0
    assert r == 1.0
    assert f == 1.0


def test_singleton_false_match():
    """Singleton with false positive match should score 0.0."""
    p, r, f = compute_entity_f05(set(), {"S2-001"})
    assert p == 0.0
    assert r == 0.0
    assert f == 0.0


def test_entity_with_matches_predicted_empty():
    """Entity with true matches but predicted empty should score 0.0."""
    p, r, f = compute_entity_f05({"S2-001"}, set())
    assert p == 0.0
    assert r == 0.0
    assert f == 0.0


def test_f05_precision_weighting():
    """Verify precision carries 2x weight over recall in F_0.5 formula."""
    # 2 true matches: S2-00047, S3-00812
    # 3 predictions: S2-00047, S2-00193, S3-00812
    # Precision = 2/3, Recall = 1.0 -> F0.5 = 0.71428...
    true_set = {"S2-00047", "S3-00812"}
    pred_set = {"S2-00047", "S2-00193", "S3-00812"}
    p, r, f = compute_entity_f05(true_set, pred_set)
    assert abs(p - 2/3) < 1e-4
    assert abs(r - 1.0) < 1e-4
    assert abs(f - 0.7143) < 1e-3


def test_evaluate_predictions_macro_average():
    """Verify macro-averaging across multiple entities."""
    gt_df = pd.DataFrame([
        {"source1_entity_id": "S1-001", "matched_entity_ids": "S2-001"},
        {"source1_entity_id": "S1-002", "matched_entity_ids": ""},  # singleton
        {"source1_entity_id": "S1-003", "matched_entity_ids": "S3-001,S3-002"}
    ])
    pred_df = pd.DataFrame([
        {"source1_entity_id": "S1-001", "matched_entity_ids": "S2-001"},  # score 1.0
        {"source1_entity_id": "S1-002", "matched_entity_ids": ""},        # score 1.0
        {"source1_entity_id": "S1-003", "matched_entity_ids": ""}         # score 0.0
    ])
    metrics = evaluate_predictions(gt_df, pred_df)
    assert abs(metrics["macro_f05"] - (1.0 + 1.0 + 0.0) / 3) < 1e-4
