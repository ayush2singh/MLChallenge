# Requirements & Boundaries

## Functional Requirements
- **R1. Ingestion**: Stream 50–100 GB TSV datasets using Polars LazyFrames / out-of-core chunking under < 10 GB RAM.
- **R2. Blocking**: High-recall candidate generation partitioned by country with multi-token / n-gram inverted indexing, bounding candidate pool to $\le 50$ pairs per S1 entity. Outputs `output/candidate_pairs.tsv`.
- **R3. Feature Engineering**: Fast vectorizable pairwise string & address similarity metrics (Jaro-Winkler, Levenshtein, token set ratio, address number match).
- **R4. Modeling & Calibration**: LightGBM binary classifier with **Brutally Strict Confidence & Singleton Gate (BSCSG)**. Default state is Singleton ($\emptyset$). The model only predicts a match if confidence exceeds a high, precision-calibrated threshold $\tau^*$ (typically $\ge 0.82$) and passes address sanity vetoes. If uncertain, it walks away and leaves the match blank. Outputs `output/matching_results.tsv`.
- **R5. Integrity Verification**: Validate outputs using `utils/validate_submission.py` ensuring zero duplicate IDs, strict subset rule, and 100% test S1 coverage.
- **R6. Delivery**: Self-contained Kaggle notebook (`business_entity_resolution_kaggle.ipynb`), runnable local package in `code/business_entity_resolution/`, and `Documentation_template.md`.

## Non-Functional Requirements & Constraints
- **C1. Strict Fair-Play**: ZERO external API lookups, geocoding, or web augmentation.
- **C2. Model License & Size**: MIT/Apache 2.0, $\le 8\text{B}$ parameters.
- **C3. Execution Limit**: Kaggle 12-hour session limit and 20 GB disk output limit.
- **C4. Timeline**: 72 hours end-to-end.
