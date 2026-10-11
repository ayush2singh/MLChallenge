# Roadmap: Large-Scale Entity Resolution

## Phases

### Phase 1: Planning & Specification (COMPLETE)
- [x] Analyze problem statement (`MLCProblem.md`)
- [x] Assess compute & memory limits (8GB local, Kaggle 30GB, 72h window)
- [x] Finalize architectural trade-offs with user
- [x] Publish `project_spec.md` and `.planning/` baseline

### Phase 2: Evaluation Suite & Local Synthetic Benchmark (COMPLETE)
- [x] Implement exact Macro $F_{0.5}$ evaluation logic and validation rules (`utils/metrics.py`)
- [x] Implement synthetic data generator mimicking US, India, and France distributions with real noise patterns (`utils/synthetic_data_generator.py`)
- [x] Implement official submission validator (`utils/validate_submission.py`)
- [x] Establish deterministic baseline test suite (9/9 unit tests passing across metrics & validator)

### Phase 3: Out-of-Core Blocking & Candidate Generation (COMPLETE)
- [x] Implement streaming TSV reader and text normalizer (`src/preprocess.py`)
- [x] Implement country-partitioned inverted index & TF-IDF candidate generation (`src/blocking.py`)
- [x] Verify candidate recall ceiling (99.48% achieved on real ground truth, >98% target) and candidate set export (`output/candidate_pairs.tsv`)
- [x] Comprehensive test suite established (22/22 unit tests passing across preprocess, blocking, metrics, and validator)

### Phase 4: Feature Engineering & LightGBM Matching (COMPLETE)
- [x] Implement high-speed pairwise ER feature extraction (`src/features.py`)
- [x] Train LightGBM classifier on candidate pairs with GroupKFold CV (`src/model.py`)
- [x] Implement Brutally Strict Confidence & Singleton Gate (BSCSG):
  - Calibrate high precision threshold $\tau^*$ directly on Macro $F_{0.5}$ (achieved $\tau^* \approx 0.760$, Macro $F_{0.5} = 0.9510$, Precision $= 0.9768$, Recall $= 0.9034$)
  - Add hard numeric/address discrepancy veto (conflicting postal codes and non-overlapping door numbers)
  - Strict "Walk Away" default: if $\max(\text{score}) < \tau^*$, leave match blank (score 1.0 on singletons)
- [x] Generate final matches (`output/matching_results.tsv`) and validated with `utils/validate_submission.py`
- [x] Comprehensive test suite established (32/32 unit tests passing across all components)

### Phase 5: Kaggle Standalone Pipeline & Submission Packaging (COMPLETE)
- [x] Build 1-click standalone Kaggle Notebook (`business_entity_resolution_kaggle.ipynb`)
- [x] Assemble `code/business_entity_resolution/` with `README.md` and pinned `requirements.txt`
- [x] Write comprehensive technical report in `Documentation_template.md`
- [x] Package `MLChallenge_Team_submission.zip` and run final validator (0 errors, 0 warnings, exit code 0)
- [x] 100% test pass rate across all 32 unit tests with zero regressions
