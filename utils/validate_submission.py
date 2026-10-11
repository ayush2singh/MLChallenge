#!/usr/bin/env python3
"""
Submission validation script for the Business Entity Resolution Challenge.
Standard library only (no external dependencies).

Checks:
1. File existence and tab-separated format for matching_results.tsv and candidate_pairs.tsv.
2. Exact header columns:
   - matching_results.tsv: source1_entity_id, matched_entity_ids
   - candidate_pairs.tsv: source1_entity_id, candidate_entity_ids
3. Completeness & Uniqueness:
   - Every Source 1 entity from test_source1.tsv must appear exactly once.
   - No duplicate source1_entity_id rows.
   - No unknown source1_entity_id values.
4. Allowed Target IDs:
   - All predicted and candidate IDs must start with 'S2-' or 'S3-'.
   - All referenced IDs must exist in test_source2.tsv or test_source3.tsv.
   - No self-matches ('S1-').
5. Duplicate checks within lists:
   - No duplicate IDs within a single comma-separated list.
6. Subset rule:
   - For every Source 1 entity, matched_entity_ids must be a subset of candidate_entity_ids.
"""

import sys
import os
import argparse
import csv
from typing import List, Set, Dict


def load_ids_from_tsv(file_path: str, id_col_name: str = "entity_id") -> Set[str]:
    """Loads a set of entity IDs from the specified column of a TSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing required file: {file_path}")
    
    ids: Set[str] = set()
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header:
            raise ValueError(f"File {file_path} is empty.")
            
        try:
            col_idx = header.index(id_col_name)
        except ValueError:
            # Fallback to column 0 if entity_id header is slightly different or missing
            col_idx = 0
            
        for line_num, row in enumerate(reader, start=2):
            if not row or len(row) <= col_idx:
                continue
            e_id = row[col_idx].strip()
            if e_id:
                ids.add(e_id)
    return ids


def validate_submission(
    matching_file: str,
    candidate_file: str,
    test_dir: str
) -> List[str]:
    errors: List[str] = []
    
    # 1. Check file existence
    if not os.path.exists(matching_file):
        return [f"Matching file not found: {matching_file}"]
    if not os.path.exists(candidate_file):
        return [f"Candidate file not found: {candidate_file}"]
        
    s1_test_path = os.path.join(test_dir, "test_source1.tsv")
    s2_test_path = os.path.join(test_dir, "test_source2.tsv")
    s3_test_path = os.path.join(test_dir, "test_source3.tsv")
    
    for path, name in [(s1_test_path, "test_source1.tsv"),
                       (s2_test_path, "test_source2.tsv"),
                       (s3_test_path, "test_source3.tsv")]:
        if not os.path.exists(path):
            errors.append(f"Required test file not found in test-dir: {name} (expected at {path})")
            
    if errors:
        return errors
        
    try:
        valid_s1_ids = load_ids_from_tsv(s1_test_path)
        valid_s2_ids = load_ids_from_tsv(s2_test_path)
        valid_s3_ids = load_ids_from_tsv(s3_test_path)
        valid_target_ids = valid_s2_ids.union(valid_s3_ids)
    except Exception as e:
        return [f"Error reading test set files: {str(e)}"]

    # 2. Parse candidate_pairs.tsv
    candidate_map: Dict[str, Set[str]] = {}
    candidate_s1_seen: Set[str] = set()
    
    with open(candidate_file, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header or len(header) < 2:
            errors.append(f"{candidate_file}: Missing or invalid tab-separated header. Expected: source1_entity_id\\tcandidate_entity_ids")
        elif header[0].strip() != "source1_entity_id" or header[1].strip() != "candidate_entity_ids":
            errors.append(f"{candidate_file}: Incorrect header columns {header}. Expected: ['source1_entity_id', 'candidate_entity_ids']")
            
        for line_no, row in enumerate(reader, start=2):
            if not row:
                continue
            s1_id = row[0].strip()
            c_str = row[1].strip() if len(row) > 1 else ""
            
            if s1_id in candidate_s1_seen:
                errors.append(f"{candidate_file}: Line {line_no} duplicate source1_entity_id '{s1_id}'")
            candidate_s1_seen.add(s1_id)
            
            if s1_id not in valid_s1_ids:
                errors.append(f"{candidate_file}: Line {line_no} unknown source1_entity_id '{s1_id}' (not in test_source1.tsv)")
                
            c_ids = [cid.strip() for cid in c_str.split(",") if cid.strip()]
            if len(c_ids) != len(set(c_ids)):
                errors.append(f"{candidate_file}: Line {line_no} contains duplicate IDs in candidate list for '{s1_id}'")
                
            for cid in c_ids:
                if cid.startswith("S1-"):
                    errors.append(f"{candidate_file}: Line {line_no} self-match illegal candidate ID '{cid}'")
                elif not (cid.startswith("S2-") or cid.startswith("S3-")):
                    errors.append(f"{candidate_file}: Line {line_no} candidate ID '{cid}' must start with 'S2-' or 'S3-'")
                elif cid not in valid_target_ids:
                    errors.append(f"{candidate_file}: Line {line_no} candidate ID '{cid}' does not exist in test_source2.tsv or test_source3.tsv")
                    
            candidate_map[s1_id] = set(c_ids)

    # 3. Parse matching_results.tsv
    matching_map: Dict[str, Set[str]] = {}
    matching_s1_seen: Set[str] = set()
    
    with open(matching_file, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header or len(header) < 2:
            errors.append(f"{matching_file}: Missing or invalid tab-separated header. Expected: source1_entity_id\\tmatched_entity_ids")
        elif header[0].strip() != "source1_entity_id" or header[1].strip() != "matched_entity_ids":
            errors.append(f"{matching_file}: Incorrect header columns {header}. Expected: ['source1_entity_id', 'matched_entity_ids']")
            
        for line_no, row in enumerate(reader, start=2):
            if not row:
                continue
            s1_id = row[0].strip()
            m_str = row[1].strip() if len(row) > 1 else ""
            
            if s1_id in matching_s1_seen:
                errors.append(f"{matching_file}: Line {line_no} duplicate source1_entity_id '{s1_id}'")
            matching_s1_seen.add(s1_id)
            
            if s1_id not in valid_s1_ids:
                errors.append(f"{matching_file}: Line {line_no} unknown source1_entity_id '{s1_id}' (not in test_source1.tsv)")
                
            m_ids = [mid.strip() for mid in m_str.split(",") if mid.strip()]
            if len(m_ids) != len(set(m_ids)):
                errors.append(f"{matching_file}: Line {line_no} contains duplicate IDs in matched list for '{s1_id}'")
                
            for mid in m_ids:
                if mid.startswith("S1-"):
                    errors.append(f"{matching_file}: Line {line_no} self-match illegal matched ID '{mid}'")
                elif not (mid.startswith("S2-") or mid.startswith("S3-")):
                    errors.append(f"{matching_file}: Line {line_no} matched ID '{mid}' must start with 'S2-' or 'S3-'")
                elif mid not in valid_target_ids:
                    errors.append(f"{matching_file}: Line {line_no} matched ID '{mid}' does not exist in test_source2.tsv or test_source3.tsv")
                    
            matching_map[s1_id] = set(m_ids)

    # 4. Check coverage against test_source1.tsv
    missing_in_matching = valid_s1_ids - matching_s1_seen
    if missing_in_matching:
        errors.append(f"{matching_file}: Missing {len(missing_in_matching)} Source 1 entities from test_source1.tsv (e.g. {list(missing_in_matching)[:5]})")
        
    missing_in_candidates = valid_s1_ids - candidate_s1_seen
    if missing_in_candidates:
        errors.append(f"{candidate_file}: Missing {len(missing_in_candidates)} Source 1 entities from test_source1.tsv (e.g. {list(missing_in_candidates)[:5]})")
        
    # 5. Check subset condition: matched_entity_ids must be a subset of candidate_entity_ids
    for s1_id, m_set in matching_map.items():
        c_set = candidate_map.get(s1_id, set())
        invalid_subset = m_set - c_set
        if invalid_subset:
            errors.append(
                f"Subset violation for entity '{s1_id}': matched IDs {list(invalid_subset)} were never present in candidate_pairs.tsv"
            )

    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate challenge submission files.")
    parser.add_argument("--matching", required=True, help="Path to output/matching_results.tsv")
    parser.add_argument("--candidate", required=True, help="Path to output/candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, help="Path to dataset/test directory")
    args = parser.parse_args()
    
    issues = validate_submission(args.matching, args.candidate, args.test_dir)
    
    if not issues:
        print("PASS")
        sys.exit(0)
    else:
        print(f"FAILED with {len(issues)} issue(s):")
        # Print up to first 25 issues
        for idx, issue in enumerate(issues[:25], start=1):
            print(f"  {idx}. {issue}")
        if len(issues) > 25:
            print(f"  ... and {len(issues) - 25} more issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()
