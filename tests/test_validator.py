"""
Unit tests for utils/validate_submission.py
"""

import os
import tempfile
import pytest
import csv
from utils.validate_submission import validate_submission


@pytest.fixture
def mock_competition_dir(tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    
    with open(test_dir / "test_source1.tsv", "w", newline="", encoding="utf-8") as f:
        f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        f.write("S1-001\tAlpha Corp\t100 Main St\tUS\n")
        f.write("S1-002\tBeta LLC\t200 Oak Ave\tUS\n")
        f.write("S1-003\tGamma SAS\t10 Rue Rivoli\tFrance\n")
        
    with open(test_dir / "test_source2.tsv", "w", newline="", encoding="utf-8") as f:
        f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        f.write("S2-001\tAlpha Corporation\t100 Main Street\tUS\n")
        
    with open(test_dir / "test_source3.tsv", "w", newline="", encoding="utf-8") as f:
        f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        f.write("S3-001\tBeta Inc\t200 Oak Avenue\tUS\n")
        f.write("S3-002\tGamma\t10 Rue Rivoli\tFrance\n")
        
    return {
        "test_dir": str(test_dir),
        "out_dir": str(out_dir),
        "cand_path": str(out_dir / "candidate_pairs.tsv"),
        "match_path": str(out_dir / "matching_results.tsv"),
    }


def test_validator_passes_valid_submission(mock_competition_dir):
    cand_path = mock_competition_dir["cand_path"]
    match_path = mock_competition_dir["match_path"]
    test_dir = mock_competition_dir["test_dir"]
    
    with open(cand_path, "w", newline="", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        f.write("S1-001\tS2-001\n")
        f.write("S1-002\tS3-001\n")
        f.write("S1-003\tS3-002\n")
        
    with open(match_path, "w", newline="", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.write("S1-001\tS2-001\n")
        f.write("S1-002\tS3-001\n")
        f.write("S1-003\t\n")  # singleton
        
    issues = validate_submission(match_path, cand_path, test_dir)
    assert issues == []


def test_validator_fails_missing_entity(mock_competition_dir):
    cand_path = mock_competition_dir["cand_path"]
    match_path = mock_competition_dir["match_path"]
    test_dir = mock_competition_dir["test_dir"]
    
    # Missing S1-003
    with open(cand_path, "w", newline="", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        f.write("S1-001\tS2-001\n")
        f.write("S1-002\tS3-001\n")
        
    with open(match_path, "w", newline="", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.write("S1-001\tS2-001\n")
        f.write("S1-002\tS3-001\n")
        
    issues = validate_submission(match_path, cand_path, test_dir)
    assert any("missing" in issue.lower() for issue in issues)


def test_validator_fails_subset_violation(mock_competition_dir):
    cand_path = mock_competition_dir["cand_path"]
    match_path = mock_competition_dir["match_path"]
    test_dir = mock_competition_dir["test_dir"]
    
    with open(cand_path, "w", newline="", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        f.write("S1-001\t\n")  # candidate is empty
        f.write("S1-002\tS3-001\n")
        f.write("S1-003\tS3-002\n")
        
    with open(match_path, "w", newline="", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.write("S1-001\tS2-001\n")  # matched ID not in candidate list!
        f.write("S1-002\tS3-001\n")
        f.write("S1-003\tS3-002\n")
        
    issues = validate_submission(match_path, cand_path, test_dir)
    assert any("subset" in issue.lower() for issue in issues)
