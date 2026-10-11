# Phase 5 Context: Kaggle Standalone Pipeline & Submission Packaging

**Phase Target**: Phase 5 (Kaggle Standalone Pipeline & Submission Packaging)  
**Parent Milestone**: Business Entity Resolution Challenge  
**Status**: Decided & Ready for Planning (`/gsd-plan-phase 5`)

---

## 1. Executive Summary & Phase Goal

Phase 5 delivers the final end-to-end deliverables for the Amazon ML Challenge 2026:
1. A clean, self-contained runnable code package in `code/business_entity_resolution/`.
2. A 1-click standalone Kaggle Notebook (`business_entity_resolution_kaggle.ipynb`) optimized for Kaggle's environment (30 GB RAM, CPU/GPU, 12-hour limit).
3. The comprehensive methodology write-up in `Documentation_template.md`.
4. The verified final competition submission archive: `MLChallenge_Team_submission.zip`.

---

## 2. Locked Implementation Decisions

Following structured architectural discovery with the user, the following design choices are finalized and locked:

### Decision 1: Submission Archive Name & Directory Structure
- **Target Archive**: `MLChallenge_Team_submission.zip`
- **Internal Structure**:
  ```text
  MLChallenge_Team_submission.zip
  ├── output/
  │   ├── matching_results.tsv       # Final matches (uploaded to leaderboard)
  │   └── candidate_pairs.tsv        # Candidate blocking set (validated subset)
  ├── code/
  │   └── business_entity_resolution/
  │       ├── src/
  │       │   ├── __init__.py
  │       │   ├── preprocess.py      # Unicode NFKD, suffix stripping, address tokens
  │       │   ├── blocking.py        # Country-partitioned dual-layer blocking
  │       │   ├── features.py        # RapidFuzz string & address feature extraction
  │       │   └── model.py           # LightGBM classifier & BSCSG inference
  │       ├── README.md              # Exact reproduction instructions
  │       └── requirements.txt       # Pinned dependencies
  └── Documentation_template.md      # In-depth technical methodology report
  ```

### Decision 2: Kaggle Notebook Architecture — Self-Contained Monolithic Notebook
- **Target File**: `business_entity_resolution_kaggle.ipynb`
- **Execution Strategy**:
  - Inlines all modular logic into self-contained, clean notebook cells so anyone can run the notebook end-to-end in 1-click on Kaggle without local file dependencies.
  - Automatically detects environment paths (`/kaggle/input/business-entity-resolution-challenge` vs local `./dataset`).
  - Writes outputs directly to `/kaggle/working/output/candidate_pairs.tsv` and `/kaggle/working/output/matching_results.tsv`.
  - Includes progress reporting, timing benchmarks, and the official verification hook.

### Decision 3: Methodology Document — High-Depth Technical Report
- **Target File**: `Documentation_template.md`
- **Required Sections**:
  1. **Executive Summary & Mathematical Formulation**: Macro $F_{0.5}$ metric dynamics, $2\times$ precision penalty, singleton mathematical dominance.
  2. **Data Ingestion & Multilingual Preprocessing**: NFKD normalization for French diacritics, multilingual legal suffix stripping across US, India, and France.
  3. **Candidate Generation / Dual-Layer Blocking**:
     - Hard country partitioning (open set).
     - Layer 1: Multi-token inverted index on core name words and postal/PIN codes.
     - Layer 2: Sparse character 2–4 n-gram TF-IDF cosine retrieval with chunked batching.
     - Top-30 candidate budget and 99.48% recall ceiling.
  4. **Pairwise Feature Engineering**: C-accelerated RapidFuzz metrics (Jaro-Winkler, Token Sort, Token Set, Levenshtein), tri-state postal matching, door number Jaccard overlap, source indicators.
  5. **Model Architecture & BSCSG Gate**:
     - Stratified 1:4 hard negative sampling.
     - LightGBM binary classifier with GroupKFold cross-validation.
     - Precision-calibrated threshold $\tau^* = 0.760$.
     - Hard postal code and door number discrepancy vetoes.
     - Strict walk-away default on singletons.
  6. **Experimental Results**: Cross-validation tables (Macro $F_{0.5} = 0.9510$, Precision $= 0.9768$, Recall $= 0.9034$), feature importances, and conclusions.

### Decision 4: Packaging & Validation Automation
- **Target Module**: `utils/package_submission.py`
  - Automated python script that assembles `code/business_entity_resolution/`, copies `output/`, embeds `Documentation_template.md`, creates `MLChallenge_Team_submission.zip`, and runs [utils/validate_submission.py](file:///c:/Users/AYUSH/Documents/MLChallenge/utils/validate_submission.py) on the archive contents.

---

## 3. Definition of Done (Phase 5 Gate)

- [ ] `code/business_entity_resolution/` assembled with `src/`, `README.md`, and pinned `requirements.txt`.
- [ ] `business_entity_resolution_kaggle.ipynb` created, tested, and validated.
- [ ] `Documentation_template.md` fully completed with deep technical analysis.
- [ ] `utils/package_submission.py` creates `MLChallenge_Team_submission.zip`.
- [ ] Final submission archive verified with `utils/validate_submission.py` (0 errors, 0 warnings, exit code 0).
- [ ] All 32 existing unit tests passing without regression.
