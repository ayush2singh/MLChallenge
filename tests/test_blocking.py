"""
Unit tests for src/blocking.py
"""

import os
import tempfile
import pytest
import pandas as pd
from src.blocking import BlockingEngine, export_candidate_pairs, run_blocking_pipeline
from utils.validate_submission import validate_submission


@pytest.fixture
def sample_data():
    """Generates small multi-country sample datasets for testing blocking behavior."""
    s1_data = [
        {"entity_id": "S1-001", "business_name": "Alpha Apex Technologies Inc.", "business_address": "100 Innovation Way, San Jose, CA 95134", "country": "US"},
        {"entity_id": "S1-002", "business_name": "Bharat Dynamic Systems Pvt. Ltd.", "business_address": "Plot 42, Electronics City, Bangalore 560100", "country": "India"},
        {"entity_id": "S1-003", "business_name": "Atelier Lumiere Artisanat SAS", "business_address": "24 Rue de Rivoli, 75004 Paris", "country": "France"},
        {"entity_id": "S1-004", "business_name": "Singleton Solitary Corp", "business_address": "999 Faraway Place, Nowhere 00000", "country": "US"},
    ]
    
    s2_data = [
        # Match for S1-001 with abbreviation & typo
        {"entity_id": "S2-101", "business_name": "Alpha Apex Tech Corp", "business_address": "100 Innovation Way, Suite 300, San Jose 95134", "country": "US"},
        # Match for S1-002
        {"entity_id": "S2-102", "business_name": "Bharat Dynamic Systems Ltd.", "business_address": "Electronics City Phase 1, Bangalore 560100", "country": "India"},
    ]
    
    s3_data = [
        # Another match for S1-001
        {"entity_id": "S3-201", "business_name": "Alpha Apex Technologies", "business_address": "San Jose, CA 95134", "country": "US"},
        # Match for S1-003 with French diacritics
        {"entity_id": "S3-202", "business_name": "Atelier Lumière Artisanat", "business_address": "24 Rue de Rivoli, Paris 75004", "country": "France"},
        # Distractor in US
        {"entity_id": "S3-203", "business_name": "Zeta Motors Corporation", "business_address": "500 Elm St, Dallas, TX 75201", "country": "US"},
    ]
    
    return {
        "df_s1": pd.DataFrame(s1_data),
        "df_s2": pd.DataFrame(s2_data),
        "df_s3": pd.DataFrame(s3_data),
    }


def test_blocking_recall_and_country_isolation(sample_data):
    """Verify high-recall retrieval and strict country isolation."""
    df_s1 = sample_data["df_s1"]
    cand_df = pd.concat([sample_data["df_s2"], sample_data["df_s3"]], ignore_index=True)
    
    blocker = BlockingEngine(top_k=10)
    candidates = blocker.generate_candidates(df_s1, cand_df)
    
    # S1-001 (US) should retrieve S2-101 and S3-201
    assert "S1-001" in candidates
    assert "S2-101" in candidates["S1-001"]
    assert "S3-201" in candidates["S1-001"]
    # US record should NEVER retrieve Indian or French records
    assert "S2-102" not in candidates["S1-001"]
    assert "S3-202" not in candidates["S1-001"]
    
    # S1-002 (India) should retrieve S2-102
    assert "S1-002" in candidates
    assert "S2-102" in candidates["S1-002"]
    assert "S2-101" not in candidates["S1-002"]
    
    # S1-003 (France) should retrieve S3-202 despite accents
    assert "S1-003" in candidates
    assert "S3-202" in candidates["S1-003"]
    
    # S1-004 (Singleton) should have 0 matches or only low-scoring US candidates, strictly <= top_k
    assert len(candidates["S1-004"]) <= 10


def test_candidate_budget_cap(sample_data):
    """Verify that candidates are strictly capped at top_k."""
    # Generate 50 mock candidate records in US
    records = []
    for i in range(50):
        records.append({
            "entity_id": f"S2-{i:03d}",
            "business_name": f"Alpha Variant {i} Inc.",
            "business_address": "100 Innovation Way, San Jose 95134",
            "country": "US"
        })
    big_cand_df = pd.DataFrame(records)
    
    df_s1 = sample_data["df_s1"].iloc[:1]  # Only S1-001
    
    top_k_budget = 15
    blocker = BlockingEngine(top_k=top_k_budget)
    candidates = blocker.generate_candidates(df_s1, big_cand_df)
    
    assert len(candidates["S1-001"]) == top_k_budget


def test_export_and_validator_compliance(sample_data, tmp_path):
    """Verify that candidate export passes the official submission validator rules."""
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    
    s1_path = str(test_dir / "test_source1.tsv")
    s2_path = str(test_dir / "test_source2.tsv")
    s3_path = str(test_dir / "test_source3.tsv")
    candidate_path = str(out_dir / "candidate_pairs.tsv")
    matching_path = str(out_dir / "matching_results.tsv")
    
    sample_data["df_s1"].to_csv(s1_path, sep="\t", index=False)
    sample_data["df_s2"].to_csv(s2_path, sep="\t", index=False)
    sample_data["df_s3"].to_csv(s3_path, sep="\t", index=False)
    
    candidates = run_blocking_pipeline(s1_path, s2_path, s3_path, candidate_path, top_k=30)
    
    # Write a matching file that is a valid subset of candidates
    with open(matching_path, "w", newline="", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in sorted(candidates.keys()):
            cands = candidates[s1_id]
            matched = cands[0] if cands else ""
            f.write(f"{s1_id}\t{matched}\n")
            
    # Validate using official validator
    issues = validate_submission(matching_path, candidate_path, str(test_dir))
    assert issues == [], f"Validation failed with issues: {issues}"
