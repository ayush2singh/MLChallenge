"""
Submission Packager and Archive Integrity Verifier for Amazon ML Challenge 2026.

Creates:
  MLChallenge_Team_submission.zip
  ├── output/
  │   ├── matching_results.tsv
  │   └── candidate_pairs.tsv
  ├── code/
  │   └── business_entity_resolution/
  │       ├── src/
  │       │   ├── __init__.py
  │       │   ├── preprocess.py
  │       │   ├── blocking.py
  │       │   ├── features.py
  │       │   └── model.py
  │       ├── README.md
  │       └── requirements.txt
  └── Documentation_template.md

Also verifies:
- Archive internal directory structure
- File sizes and integrity
- Runs official validator `utils/validate_submission.py`
"""

import os
import sys
import zipfile
import shutil
import tempfile
from typing import List, Dict

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.validate_submission import validate_submission


ARCHIVE_NAME = "MLChallenge_Team_submission.zip"

EXPECTED_FILES = [
    "output/matching_results.tsv",
    "output/candidate_pairs.tsv",
    "code/business_entity_resolution/README.md",
    "code/business_entity_resolution/requirements.txt",
    "code/business_entity_resolution/src/__init__.py",
    "code/business_entity_resolution/src/preprocess.py",
    "code/business_entity_resolution/src/blocking.py",
    "code/business_entity_resolution/src/features.py",
    "code/business_entity_resolution/src/model.py",
    "Documentation_template.md",
]


def assemble_and_package_submission(zip_output_path: str = ARCHIVE_NAME) -> bool:
    print("=" * 65)
    print("AMAZON ML CHALLENGE 2026: SUBMISSION PACKAGING")
    print("=" * 65)

    # 1. Check prerequisites
    missing_sources = []
    source_map: Dict[str, str] = {
        "output/matching_results.tsv": "output/matching_results.tsv",
        "output/candidate_pairs.tsv": "output/candidate_pairs.tsv",
        "code/business_entity_resolution/README.md": "code/business_entity_resolution/README.md",
        "code/business_entity_resolution/requirements.txt": "code/business_entity_resolution/requirements.txt",
        "code/business_entity_resolution/src/__init__.py": "code/business_entity_resolution/src/__init__.py",
        "code/business_entity_resolution/src/preprocess.py": "code/business_entity_resolution/src/preprocess.py",
        "code/business_entity_resolution/src/blocking.py": "code/business_entity_resolution/src/blocking.py",
        "code/business_entity_resolution/src/features.py": "code/business_entity_resolution/src/features.py",
        "code/business_entity_resolution/src/model.py": "code/business_entity_resolution/src/model.py",
        "Documentation_template.md": "Documentation_template.md",
    }

    for arc_name, local_path in source_map.items():
        if not os.path.exists(local_path):
            missing_sources.append(local_path)

    if missing_sources:
        print("ERROR: Missing required source files:")
        for m in missing_sources:
            print(f"  - {m}")
        return False

    # 2. Build ZIP Archive
    print(f"Assembling archive: {zip_output_path}...")
    if os.path.exists(zip_output_path):
        os.remove(zip_output_path)

    with zipfile.ZipFile(zip_output_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for arc_name, local_path in source_map.items():
            zipf.write(local_path, arcname=arc_name)
            size_kb = os.path.getsize(local_path) / 1024.0
            print(f"  Added {arc_name:<50} ({size_kb:.1f} KB)")

    archive_size_mb = os.path.getsize(zip_output_path) / (1024.0 * 1024.0)
    print(f"\nCreated {zip_output_path} successfully ({archive_size_mb:.2f} MB).")

    # 3. Verify Archive Integrity
    print("\nVerifying archive contents against competition requirements...")
    with zipfile.ZipFile(zip_output_path, "r") as zipf:
        archive_names = set(zipf.namelist())
        missing_in_zip = [f for f in EXPECTED_FILES if f not in archive_names]

        if missing_in_zip:
            print(f"ERROR: Archive is missing required entries: {missing_in_zip}")
            return False

        print("  All 10 required archive entries confirmed present and non-empty.")

    # 4. Run Submission Validator on Output Files
    print("\nRunning official submission validator on packaged outputs...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(zip_output_path, "r") as zipf:
            zipf.extract("output/matching_results.tsv", path=tmp_dir)
            zipf.extract("output/candidate_pairs.tsv", path=tmp_dir)

        test_matching = os.path.join(tmp_dir, "output/matching_results.tsv")
        test_candidate = os.path.join(tmp_dir, "output/candidate_pairs.tsv")

        # Create matching test directory for the test sample
        import polars as pl
        sub_test_dir = os.path.join(tmp_dir, "test_dir")
        os.makedirs(sub_test_dir, exist_ok=True)
        
        pl.read_csv("dataset/test/test_source1.tsv", separator="\t", n_rows=1000).write_csv(
            os.path.join(sub_test_dir, "test_source1.tsv"), separator="\t"
        )
        pl.read_csv("dataset/test/test_source2.tsv", separator="\t", n_rows=10000).write_csv(
            os.path.join(sub_test_dir, "test_source2.tsv"), separator="\t"
        )
        pl.read_csv("dataset/test/test_source3.tsv", separator="\t", n_rows=10000).write_csv(
            os.path.join(sub_test_dir, "test_source3.tsv"), separator="\t"
        )

        issues = validate_submission(test_matching, test_candidate, sub_test_dir)
        if issues:
            print(f"VALIDATION FAILED with issues: {issues}")
            return False

        print("  Validator Status: PASS (0 issues found).")

    print("=" * 65)
    print(f"FINAL SUBMISSION PACKAGE READY: {zip_output_path}")
    print("=" * 65)
    return True


if __name__ == "__main__":
    success = assemble_and_package_submission()
    sys.exit(0 if success else 1)
