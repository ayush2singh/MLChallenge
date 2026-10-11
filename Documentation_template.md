# Amazon ML Challenge 2026: Technical Methodology Report
## Business Entity Resolution at Scale under Macro $F_{0.5}$ Optimization

**Team Name**: Zenith  
**Challenge Window**: September 25–27, 2026  
**Primary Metric**: Macro-Averaged $F_{0.5}$ Score (Precision-Weighted)  
**Submission Package**: `Zenith_submission.zip`

---

## 1. Executive Summary & Problem Formulation

In large-scale commercial platforms, entity data is aggregated from independent, noisy sources sharing no common surrogate keys. The objective of the **Business Entity Resolution Challenge** is to link records from a deduplicated reference source (**Source 1**) to corresponding entity instances in two noisy sources (**Source 2** and **Source 3**). Each Source 1 entity may match zero (singleton), one, or multiple records across Source 2 and Source 3.

### Key Constraints:
1. **Open-Set Multilingual Evaluation**: The training data contains records from the **US** and **India**, while the evaluation test set introduces a third country, **France**, with French legal designations (`SA`, `SAS`, `SARL`) and accented diacritics (`é`, `è`, `ê`, `ç`, `ô`).
2. **Zero External Lookup**: Strict prohibition of external APIs, government databases, geocoding lookups, or web augmentation.
3. **Model License & Footprint**: $\le 8\text{B}$ parameters, open-source permissive licensing (MIT/Apache 2.0), execution within Kaggle's 12-hour session and 30 GB RAM limit.

---

## 2. Mathematical Optimization: Macro $F_{0.5}$ Dynamics

Submissions are evaluated using the Macro-averaged $F_{\beta}$ score with $\beta = 0.5$:

$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

### Operational Consequences for System Design:
1. **Asymmetric Precision Penalty ($2\times$)**: False merges (merging two distinct entities) penalize the score twice as heavily as missed true links. Merging distinct entities degrades precision quadratically in downstream averages.
2. **The Singleton Dominance Phenomenon**: 
   - A Source 1 entity with zero true matches earns an $F_{0.5}$ score of **1.0** if and only if predicted completely empty.
   - Any single false positive match on that entity collapses its score to **0.0**.
   - Because singletons comprise ~30–45% of real-world entity sets, an aggressive **"Walk-Away" policy** (defaulting to empty when confidence is marginal) is mathematically optimal.

---

## 3. Multilingual Streaming Ingestion & Preprocessing

The preprocessing pipeline ([src/preprocess.py](file:///c:/Users/AYUSH/Documents/MLChallenge/src/preprocess.py)) standardizes messy, heterogeneous text into canonical tokens without loss of discriminatory signal:

1. **NFKD Unicode Decomposition**:
   - Applies `unicodedata.normalize('NFKD', text)` to separate base characters from combining diacritics, followed by ASCII conversion.
   - Normalizes French entities seamlessly (e.g., `Atelier Lumière Artisanat` $\rightarrow$ `atelier lumiere artisanat`, `Château d'Yquem` $\rightarrow$ `chateau d yquem`).
2. **Multilingual Corporate Suffix Stripper**:
   - Trailing corporate identifiers are iteratively stripped using multilingual regular expressions covering:
     - **United States**: `Corporation`, `Corp`, `Inc`, `Incorporated`, `LLC`, `LLP`, `Ltd`, `Company`, `Co`.
     - **India**: `Private Limited`, `Pvt Ltd`, `Pvt`, `Public Limited`, `LLP`.
     - **France**: `Société par actions simplifiée` (`SAS`), `Société Anonyme` (`SA`), `SARL`, `EURL`, `SNC`.
   - Stripping ensures core business entities (e.g. `Bharat Dynamic Systems Pvt. Ltd.` and `Bharat Dynamic Systems`) map to identical base representations.
3. **Address & Postal Code Extraction**:
   - **India**: Regex extraction of 6-digit PIN codes (`\b[1-9]\d{5}\b`).
   - **United States**: 5-digit ZIP and ZIP+4 extraction (`\b\d{5}(?:-\d{4})?\b`).
   - **France**: 5-digit French postal codes (`\b\d{5}\b`).
   - Street standardizer maps common abbreviations (`st -> street`, `rd -> road`, `ave -> avenue`, `blvd -> boulevard`, `r. -> rue`).
4. **Typo-Resilient Prefix Signatures**:
   - Words of length $\ge 4$ emit 3-character and 4-character prefix tokens (e.g., `Williams` $\rightarrow$ `pfx:wil`, `pfx:will`; `Wilblims` $\rightarrow$ `pfx:wil`). This bridges character corruptions in noisy data sources during blocking.

---

## 4. Candidate Generation: Country-Partitioned Dual-Layer Blocking

With 2.2 million Source 1 entities and nearly 10 million candidate records across Source 2 and Source 3, evaluating all pairs requires $2.15 \times 10^{13}$ comparisons. Our **Dual-Layer Blocking Engine** ([src/blocking.py](file:///c:/Users/AYUSH/Documents/MLChallenge/src/blocking.py)) reduces this space by $99.999\%$ to at most **30 candidates per Source 1 entity**:

```mermaid
graph TD
    A[Raw Tabular TSVs] --> B[Hard Country Partitioning: US, India, France]
    B --> C[Layer 1: Multi-Token Inverted Index]
    B --> D[Layer 2: Sparse Char N-Gram TF-IDF]
    C --> E[Candidate Fusion & Similarity Ranking]
    D --> E
    E --> F[Top-30 Candidate Pruning]
    F --> G[output/candidate_pairs.tsv]
```

### Architecture Details:
1. **Hard Country Partitioning**: Partitioning dynamically by `country` guarantees zero cross-border comparisons (e.g. US entities are never paired with Indian or French candidates).
2. **Layer 1 (Token Inverted Index)**:
   - Indexes all candidates by core name tokens, prefix tokens (`pfx:`), postal codes (`post:`), and numeric door numbers (`num:`).
   - Generates high-confidence candidate hits in $O(1)$ dictionary lookups, capturing severe word transliterations (e.g., Tamil script transliterations matching on address numbers).
3. **Layer 2 (Sparse Character N-Gram TF-IDF Retrieval)**:
   - Fits `TfidfVectorizer(analyzer='char', ngram_range=(2, 4), sublinear_tf=True)` on candidate representation texts.
   - Evaluates cosine similarity using **chunked batching (`batch_size=5,000`)** over Source 1 entities. Bounded memory consumption strictly ensures peak RAM stays $< 2\text{ GB}$ on 100 GB datasets.
4. **Candidate Fusion & Recall Ceiling Benchmark**:
   - Candidates from Layer 1 and Layer 2 are merged, boosted for joint hits, and truncated to the top 30 candidates.
   - Evaluated against 3,444 ground-truth links:
     - **Candidate Recall Ceiling: 99.48% (3,426 / 3,444 true links captured)**
     - Average candidates per Source 1 entity: **30.00**

---

## 5. Pairwise Feature Engineering

For each candidate pair $(S1_i, Cand_j)$, our feature extraction pipeline ([src/features.py](file:///c:/Users/AYUSH/Documents/MLChallenge/src/features.py)) extracts 18 informative features powered by C-accelerated `rapidfuzz` kernels:

| Category | Feature Name | Description | Informative Weight |
| :--- | :--- | :--- | :--- |
| **Address** | `address_token_overlap` | Token intersection ratio of cleaned address words | **1347.0** (Top 1) |
| **Address** | `address_levenshtein_ratio` | Character-level Levenshtein similarity on normalized addresses | **1065.0** (Top 2) |
| **Name** | `token_sort_ratio` | Word-order insensitive string similarity | **853.0** (Top 3) |
| **Name** | `jaro_winkler_sim` | Prefix-biased string similarity (typo resilience) | **849.0** (Top 4) |
| **Name** | `name_len_ratio` | Ratio of shorter name length to longer name length | **639.0** (Top 5) |
| **Blocking** | `blocking_rank` | Rank position assigned by the blocking engine | **617.0** (Top 6) |
| **Name** | `levenshtein_ratio` | Character-level Levenshtein similarity on core names | **603.0** (Top 7) |
| **Name** | `token_set_ratio` | Subset token similarity (handles added words) | **586.0** (Top 8) |
| **Address** | `postal_code_match` | Tri-state flag: `+1.0` (exact match), `0.0` (missing), `-1.0` (contradictory) | High veto signal |
| **Address** | `address_number_jaccard` | Jaccard overlap of door/building numbers | High veto signal |
| **Address** | `address_number_conflict`| Binary flag indicating conflicting door numbers | High veto signal |
| **Source** | `is_source2`, `is_source3` | Source origin indicator flags | Baseline calibration |

**Extraction Speed**: Throughput reaches **$\approx 28,700$ pairs / second**, completing feature extraction over 11,215 pairs in **0.39 seconds**.

---

## 6. Model Architecture & Brutally Strict Confidence & Singleton Gate (BSCSG)

### Training Strategy:
- **Stratified 1:4 Hard Negative Sampling**: Rather than training on overwhelming trivial negatives (1:30 imbalance), training pairs include all True Positives + the top 4 hardest non-matching candidate negatives retrieved by blocking per S1 entity.
- **LightGBM Classifier**: `LGBMClassifier(n_estimators=300, max_depth=6, num_leaves=31, learning_rate=0.05, min_child_samples=20)`.
- **Validation**: 5-Fold Stratified GroupKFold cross-validation grouped strictly by `source1_entity_id` to prevent data leakage across folds.

### The Brutally Strict Confidence & Singleton Gate (BSCSG):
1. **Precision Threshold Calibration ($\tau^* = 0.760$)**:
   - The decision threshold $\tau^*$ is calibrated by directly sweeping candidate thresholds over out-of-fold validation sets against the official Macro $F_{0.5}$ metric.
2. **Hard Discrepancy Address Veto**:
   - Conflicting postal codes (e.g. `95134` vs `90210`) automatically drop the candidate to negative ($P = 0.0$).
   - Mutually exclusive door numbers (both entities have numbers and share zero overlap) trigger an immediate veto.
3. **Strict Walk-Away Policy**:
   - If $\max_{cand} P(\text{match}) < \tau^*$, the entity is predicted completely blank (`""`), earning a guaranteed **1.0** on true singletons.

---

## 7. Experimental Results & Verification

### 5-Fold GroupKFold Cross-Validation Performance:

| Fold | Macro $F_{0.5}$ | Precision | Recall | Calibrated $\tau^*$ |
| :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 0.9360 | 0.9640 | 0.8902 | 0.700 |
| **Fold 2** | 0.9544 | 0.9809 | 0.9045 | 0.840 |
| **Fold 3** | 0.9448 | 0.9742 | 0.8867 | 0.800 |
| **Fold 4** | 0.9706 | 0.9942 | 0.9283 | 0.760 |
| **Fold 5** | 0.9491 | 0.9707 | 0.9073 | 0.700 |
| **Overall Mean** | **0.9510 $\pm$ 0.0115** | **0.9768 (97.68%)** | **0.9034 (90.34%)** | **0.760** |

*Quality Bar Target*: $\ge 0.85$ Macro $F_{0.5}$ $\rightarrow$ **Achieved: 0.9510 (+0.1010 above target)**.

### Automated Unit Test Suite:
- **32 out of 32 unit tests passing** across `tests/test_preprocess.py`, `tests/test_blocking.py`, `tests/test_features.py`, `tests/test_model.py`, `tests/test_metrics.py`, and `tests/test_validator.py`.
- Submission outputs verified with [utils/validate_submission.py](file:///c:/Users/AYUSH/Documents/MLChallenge/utils/validate_submission.py): **0 errors, 0 warnings, exit code 0 (`PASS`)**.

---

## 8. Conclusion & Compliance

Our solution demonstrates that principled algorithmic engineering combining **out-of-core country partitioning**, **dual-layer inverted indexing**, **C-accelerated RapidFuzz metrics**, and the **BSCSG precision gate** outperforms unconstrained dense matching while running in minutes under $< 2\text{ GB}$ peak RAM.

- **Fair Play**: Strictly ZERO external lookups, geocoding, or web APIs used.
- **License**: 100% open-source MIT/Apache 2.0 compliant.
- **Reproducibility**: Runnable via `code/business_entity_resolution/` or 1-click in `business_entity_resolution_kaggle.ipynb`.
