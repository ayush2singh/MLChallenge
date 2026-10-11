"""
Unit tests for src/features.py
"""

import pytest
import pandas as pd
from src.features import compute_pair_features, extract_features_for_pairs, FEATURE_NAMES
from src.preprocess import preprocess_record


def test_name_features_typo_and_transposition():
    """Verify string similarity metrics on typos and word transpositions."""
    s1 = preprocess_record({
        "entity_id": "S1-001",
        "business_name": "Maure Williams Colombier Inc",
        "business_address": "85 Wayne Avenue, Ticonderoga, NY 12883",
        "country": "US"
    })
    cand_typo = preprocess_record({
        "entity_id": "S2-101",
        "business_name": "Maure Wilblims Colombier",
        "business_address": "85 Wayne Ave, Ticonderoga",
        "country": "US"
    })
    
    feats = compute_pair_features(s1, cand_typo, rank=1)
    # Jaro-Winkler should be very high despite 'wilblims' typo
    assert feats["jaro_winkler_sim"] > 0.90
    assert feats["token_sort_ratio"] > 0.85
    assert feats["exact_name_match"] == 0.0
    assert feats["is_source2"] == 1.0
    assert feats["is_source3"] == 0.0


def test_name_features_word_order_transposition():
    """Verify token sort ratio handles word transpositions gracefully."""
    s1 = preprocess_record({
        "entity_id": "S1-002",
        "business_name": "Apex Global Solutions LLC",
        "business_address": "100 Innovation Way, San Jose 95134",
        "country": "US"
    })
    cand_transposed = preprocess_record({
        "entity_id": "S3-202",
        "business_name": "Global Apex Solutions Inc",
        "business_address": "100 Innovation Way",
        "country": "US"
    })
    
    feats = compute_pair_features(s1, cand_transposed, rank=2)
    assert feats["token_sort_ratio"] == 1.0
    assert feats["is_source3"] == 1.0
    assert feats["blocking_rank"] == 2.0


def test_postal_code_tri_state():
    """Verify postal code matching gives +1.0 for match, -1.0 for conflict, 0.0 for missing."""
    s1 = preprocess_record({
        "entity_id": "S1-001",
        "business_name": "Acme Corp",
        "business_address": "Main St, San Jose 95134",
        "country": "US"
    })
    
    cand_match = preprocess_record({
        "entity_id": "S2-001",
        "business_name": "Acme Inc",
        "business_address": "100 Main St, 95134",
        "country": "US"
    })
    cand_conflict = preprocess_record({
        "entity_id": "S2-002",
        "business_name": "Acme Inc",
        "business_address": "100 Main St, 90210",
        "country": "US"
    })
    cand_missing = preprocess_record({
        "entity_id": "S2-003",
        "business_name": "Acme Inc",
        "business_address": "Main St",
        "country": "US"
    })
    
    assert compute_pair_features(s1, cand_match)["postal_code_match"] == 1.0
    assert compute_pair_features(s1, cand_conflict)["postal_code_match"] == -1.0
    assert compute_pair_features(s1, cand_missing)["postal_code_match"] == 0.0


def test_address_number_jaccard_and_conflict():
    """Verify address number overlap and conflict indicators."""
    s1 = preprocess_record({
        "entity_id": "S1-001",
        "business_name": "Nexus Center",
        "business_address": "Plot 42, Electronics City 560100",
        "country": "India"
    })
    cand_same_num = preprocess_record({
        "entity_id": "S2-001",
        "business_name": "Nexus Center",
        "business_address": "Plot 42, Phase 1, Bangalore 560100",
        "country": "India"
    })
    cand_diff_num = preprocess_record({
        "entity_id": "S2-002",
        "business_name": "Nexus Center",
        "business_address": "Plot 99, Phase 2, Bangalore 560100",
        "country": "India"
    })
    
    feats_same = compute_pair_features(s1, cand_same_num)
    assert feats_same["address_number_jaccard"] > 0.0
    assert feats_same["address_number_conflict"] == 0.0
    
    feats_diff = compute_pair_features(s1, cand_diff_num)
    # Both have door numbers (42 vs 99), sharing postal 560100
    assert feats_diff["address_number_conflict"] == 0.0  # shares 560100


def test_batch_extraction():
    """Verify batch feature extraction function."""
    s1_dict = {
        "S1-001": preprocess_record({"entity_id": "S1-001", "business_name": "Alpha Corp", "business_address": "123 Main St", "country": "US"}),
        "S1-002": preprocess_record({"entity_id": "S1-002", "business_name": "Beta LLC", "business_address": "456 Oak Rd", "country": "US"}),
    }
    cand_dict = {
        "S2-001": preprocess_record({"entity_id": "S2-001", "business_name": "Alpha Inc", "business_address": "123 Main St", "country": "US"}),
        "S3-001": preprocess_record({"entity_id": "S3-001", "business_name": "Beta Corp", "business_address": "456 Oak Road", "country": "US"}),
    }
    pairs = {
        "S1-001": ["S2-001"],
        "S1-002": ["S3-001"],
    }
    
    df, keys = extract_features_for_pairs(s1_dict, cand_dict, pairs)
    assert len(df) == 2
    assert len(keys) == 2
    assert list(df.columns) == FEATURE_NAMES
    assert keys[0] == ("S1-001", "S2-001")
    assert keys[1] == ("S1-002", "S3-001")
