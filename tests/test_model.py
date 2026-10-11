"""
Unit tests for src/model.py
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from src.preprocess import preprocess_record
from src.features import FEATURE_NAMES
from src.model import (
    EntityResolutionModel,
    build_training_dataset,
    calibrate_threshold_f05,
    predict_matches_bscsg,
    export_matching_results
)
from utils.validate_submission import validate_submission


@pytest.fixture
def mock_dataset():
    s1_dict = {
        "S1-001": preprocess_record({"entity_id": "S1-001", "business_name": "Apex Innovations Inc", "business_address": "100 Innovation Way, San Jose 95134", "country": "US"}),
        "S1-002": preprocess_record({"entity_id": "S1-002", "business_name": "Metro Healthcare LLC", "business_address": "450 Lexington Ave, New York 10017", "country": "US"}),
        "S1-003": preprocess_record({"entity_id": "S1-003", "business_name": "Lone Ranger Singleton", "business_address": "100 Lonely St, Austin 78701", "country": "US"}),
    }
    cand_dict = {
        # S1-001 matches
        "S2-101": preprocess_record({"entity_id": "S2-101", "business_name": "Apex Innovations Corp", "business_address": "100 Innovation Way, Ste 300, 95134", "country": "US"}),
        # S1-002 matches
        "S3-201": preprocess_record({"entity_id": "S3-201", "business_name": "Metro Healthcare", "business_address": "450 Lexington Ave, NY 10017", "country": "US"}),
        # Negative / Distractor for S1-001 with conflicting postal code
        "S2-999": preprocess_record({"entity_id": "S2-999", "business_name": "Apex Innovations", "business_address": "200 Different Road, Dallas 75201", "country": "US"}),
        # Negative for S1-002
        "S3-888": preprocess_record({"entity_id": "S3-888", "business_name": "Unrelated Healthcare Inc", "business_address": "100 Broadway, 10001", "country": "US"}),
    }
    candidate_pairs = {
        "S1-001": ["S2-101", "S2-999"],
        "S1-002": ["S3-201", "S3-888"],
        "S1-003": ["S2-999"],  # low quality candidate for singleton
    }
    ground_truth = {
        "S1-001": {"S2-101"},
        "S1-002": {"S3-201"},
        "S1-003": set(),  # true singleton
    }
    return {
        "s1_dict": s1_dict,
        "cand_dict": cand_dict,
        "candidate_pairs": candidate_pairs,
        "ground_truth": ground_truth,
    }


def test_build_training_dataset(mock_dataset):
    s1_dict = mock_dataset["s1_dict"]
    cand_dict = mock_dataset["cand_dict"]
    candidate_pairs = mock_dataset["candidate_pairs"]
    ground_truth = mock_dataset["ground_truth"]

    X, y, groups, pair_keys = build_training_dataset(
        s1_dict, cand_dict, candidate_pairs, ground_truth, neg_ratio=2
    )

    assert len(X) == len(y) == len(groups) == len(pair_keys)
    assert 1 in y and 0 in y
    assert set(X.columns) == set(FEATURE_NAMES)
    assert ("S1-001", "S2-101") in pair_keys
    idx_pos = pair_keys.index(("S1-001", "S2-101"))
    assert y[idx_pos] == 1


def test_model_training_and_calibration(mock_dataset):
    s1_dict = mock_dataset["s1_dict"]
    cand_dict = mock_dataset["cand_dict"]
    candidate_pairs = mock_dataset["candidate_pairs"]
    ground_truth = mock_dataset["ground_truth"]

    X, y, groups, pair_keys = build_training_dataset(
        s1_dict, cand_dict, candidate_pairs, ground_truth, neg_ratio=2
    )

    model = EntityResolutionModel(n_estimators=20, max_depth=3)
    model.fit(X, y)

    probs = model.predict_proba(X)
    assert len(probs) == len(y)
    assert all(0.0 <= p <= 1.0 for p in probs)

    best_tau, best_f05 = calibrate_threshold_f05(
        model, X, pair_keys, ground_truth, list(s1_dict.keys()), min_tau=0.50, max_tau=0.90, step=0.05
    )
    assert best_tau >= 0.50
    assert best_f05 >= 0.0


def test_bscsg_hard_discrepancy_veto(mock_dataset):
    """Verify that conflicting postal code triggers hard veto."""
    s1_dict = mock_dataset["s1_dict"]
    cand_dict = mock_dataset["cand_dict"]
    pair_keys = [("S1-001", "S2-999")]  # conflicting postal: 95134 vs 75201
    all_s1 = ["S1-001"]

    model = EntityResolutionModel(n_estimators=10)
    # Mock model that always predicts high confidence
    class MockHighConfidenceModel:
        calibrated_threshold = 0.50
        def predict_proba(self, X):
            return np.ones(len(X)) * 0.99

    preds = predict_matches_bscsg(
        MockHighConfidenceModel(),
        pd.DataFrame([{"dummy": 1}]),
        pair_keys,
        all_s1,
        s1_dict=s1_dict,
        cand_dict=cand_dict,
        apply_veto=True
    )

    # S2-999 has postal code 75201 which conflicts with S1-001's 95134 -> must be vetoed!
    assert preds["S1-001"] == []


def test_bscsg_singleton_walkaway(mock_dataset):
    """Verify singleton entities with weak candidates walk away with empty matches."""
    s1_dict = mock_dataset["s1_dict"]
    cand_dict = mock_dataset["cand_dict"]
    pair_keys = [("S1-003", "S2-999")]
    all_s1 = ["S1-003"]

    class MockLowConfidenceModel:
        calibrated_threshold = 0.85
        def predict_proba(self, X):
            return np.array([0.35])  # low confidence below tau=0.85

    preds = predict_matches_bscsg(
        MockLowConfidenceModel(),
        pd.DataFrame([{"dummy": 1}]),
        pair_keys,
        all_s1,
        s1_dict=s1_dict,
        cand_dict=cand_dict,
        apply_veto=True
    )

    # Low probability candidate must be pruned -> empty list
    assert preds["S1-003"] == []


def test_export_and_validate_matching(tmp_path):
    out_dir = tmp_path / "output"
    test_dir = tmp_path / "test"
    out_dir.mkdir()
    test_dir.mkdir()

    # Create dummy test files
    with open(test_dir / "test_source1.tsv", "w", newline="", encoding="utf-8") as f:
        f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        f.write("S1-001\tA\tB\tUS\n")
        f.write("S1-002\tC\tD\tUS\n")

    with open(test_dir / "test_source2.tsv", "w", newline="", encoding="utf-8") as f:
        f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        f.write("S2-001\tA\tB\tUS\n")

    with open(test_dir / "test_source3.tsv", "w", newline="", encoding="utf-8") as f:
        f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        f.write("S3-001\tC\tD\tUS\n")

    cand_path = str(out_dir / "candidate_pairs.tsv")
    match_path = str(out_dir / "matching_results.tsv")

    with open(cand_path, "w", newline="", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        f.write("S1-001\tS2-001\n")
        f.write("S1-002\t\n")

    matches = {"S1-001": ["S2-001"], "S1-002": []}
    export_matching_results(matches, match_path)

    issues = validate_submission(match_path, cand_path, str(test_dir))
    assert issues == []
