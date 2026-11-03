# 🏢 Business Entity Resolution at Scale

> **Amazon ML Challenge 2026** — Linking noisy business records across sources using precision-optimised entity resolution under Macro F₀.₅.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/model-LightGBM-green.svg)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution Architecture](#-solution-architecture)
- [Key Results](#-key-results)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Pipeline Deep Dive](#-pipeline-deep-dive)
  - [Stage 1: Preprocessing](#stage-1-multilingual-preprocessing)
  - [Stage 2: Blocking (Candidate Generation)](#stage-2-country-partitioned-blocking)
  - [Stage 3: Feature Engineering](#stage-3-pairwise-feature-engineering)
  - [Stage 4: Model & BSCSG](#stage-4-lightgbm--bscsg-decision-engine)
- [Evaluation & Metrics](#-evaluation--metrics)
- [Testing](#-testing)
- [Kaggle Deployment](#-kaggle-deployment)
- [Constraints & Compliance](#-constraints--compliance)
- [Team](#-team)

---

## 🎯 Problem Statement

Business entity data arrives from **three independent sources** with significant noise — abbreviations, transliterations, missing fields, and inconsistent address formats:

| Source | Role | Quality |
|--------|------|---------|
| **Source 1 (S1)** | Deduplicated reference entities | Clean |
| **Source 2 (S2)** | Noisy business records | High noise |
| **Source 3 (S3)** | Noisy business records | High noise |

**Goal:** For every entity in S1, find all matching entity IDs in S2 and S3. Each S1 entity may match **zero** (singleton), **one**, or **multiple** records.

**Critical twist:** The test set introduces **France** (unseen country) alongside training countries **US** and **India**.

### Why F₀.₅ Changes Everything

The evaluation metric is **Macro-averaged F₀.₅**, which weights precision **2× heavier** than recall:

$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

This means:
- ❌ A single false merge on a singleton drops its score from **1.0 → 0.0**
- ✅ Missing a true match only loses partial recall
- 🎯 **Conservative, high-precision matching is mathematically optimal**

---

## 🏗️ Solution Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                    RAW TSV DATA (50–100 GB)                       │
└──────────────┬────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  STAGE 1: PREPROCESSING         │
│  • NFKD Unicode normalisation   │
│  • Multilingual suffix strip    │
│  • Postal/PIN code extraction   │
│  • Prefix token signatures      │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  STAGE 2: BLOCKING              │
│  • Country partitioning         │
│  • Multi-channel inverted index │
│  • Top-50 candidate pruning     │
│  → candidate_pairs.tsv          │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  STAGE 3: FEATURE ENGINEERING   │
│  • 26 pairwise features         │
│  • Name, Address, Cross-field   │
│  • RapidFuzz C-accelerated      │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  STAGE 4: MODEL + BSCSG         │
│  • LightGBM binary classifier   │
│  • F₀.₅ threshold calibration   │
│  • Postal/address veto gates    │
│  • Walk-Away singleton policy   │
│  → matching_results.tsv         │
└──────────────────────────────────┘
```

---

## 🏆 Key Results

| Metric | Score |
|--------|-------|
| **Macro F₀.₅** | **0.9510 ± 0.0115** |
| Macro Precision | 0.9768 (97.68%) |
| Macro Recall | 0.9034 (90.34%) |
| Blocking Pair Recall | 99.48% |
| Calibrated Threshold τ* | 0.760 |
| Unit Tests | 32/32 passing ✅ |
| Submission Validation | PASS ✅ |

> Quality target was ≥ 0.85 Macro F₀.₅ — achieved **0.9510** (+0.1010 above target).

---

## 📁 Project Structure

```
MLChallenge/
├── src/                          # Core pipeline modules
│   ├── __init__.py
│   ├── preprocess.py             # Text normalisation, suffix stripping, postal extraction
│   ├── blocking.py               # Country-partitioned inverted index blocking engine
│   ├── features.py               # 26-feature pairwise extraction (RapidFuzz)
│   └── model.py                  # LightGBM classifier + BSCSG decision engine
│
├── utils/                        # Supporting utilities
│   ├── metrics.py                # Macro F₀.₅ scorer (exact competition metric)
│   ├── validate_submission.py    # 6-rule submission validator
│   ├── synthetic_data_generator.py  # Synthetic data for local testing
│   ├── train_and_evaluate.py     # Full train/eval orchestrator
│   ├── kaggle_all_in_one.py      # Kaggle notebook generator
│   ├── benchmark_blocking.py     # Blocking recall benchmarks
│   └── package_submission.py     # Submission ZIP packager
│
├── tests/                        # Comprehensive test suite
│   ├── test_preprocess.py        # Normalisation, suffix stripping, postal codes
│   ├── test_blocking.py          # Blocking engine and recall
│   ├── test_features.py          # Feature computation correctness
│   ├── test_model.py             # Model training and prediction
│   ├── test_metrics.py           # F₀.₅ scorer edge cases
│   └── test_validator.py         # Submission validation rules
│
├── code/business_entity_resolution/
│   ├── src/main.py               # Standalone entry point
│   └── requirements.txt          # Pinned dependencies
│
├── dataset/                      # Data directory (gitignored)
│   ├── train/                    # Training TSVs + ground truth
│   └── test/                     # Test TSVs
│
├── output/                       # Generated outputs (gitignored)
│   ├── candidate_pairs.tsv       # Blocking output
│   └── matching_results.tsv      # Final predictions
│
├── business_entity_resolution_kaggle.ipynb  # Self-contained Kaggle notebook
├── benchmark_local.py            # Local benchmark runner
├── Documentation_template.md     # Full technical methodology report
├── project_spec.md               # Detailed project specification
└── MLCProblem.md                 # Original problem statement
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- ~4 GB RAM (pipeline is designed for out-of-core processing)

### Installation

```bash
git clone https://github.com/ayush2singh/MLChallenge.git
cd MLChallenge
pip install -r code/business_entity_resolution/requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `polars` ≥ 1.0.0 | Streaming out-of-core TSV ingestion |
| `numpy` ≥ 1.24.0 | Numerical operations |
| `lightgbm` ≥ 4.0.0 | Gradient-boosted tree classifier |
| `rapidfuzz` ≥ 3.0.0 | C-accelerated string similarity (Jaro-Winkler, Levenshtein) |
| `unidecode` ≥ 1.3.6 | Unicode transliteration fallback |
| `pandas` | DataFrame operations |

### Run Local Benchmark

```bash
# Quick benchmark (5,000 S1 entities)
python benchmark_local.py --sample-size 5000

# Full benchmark (20,000 S1 entities)
python benchmark_local.py --sample-size 20000
```

### Validate Submission

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

### Run Tests

```bash
pytest tests/ -v
```

---

## 🔬 Pipeline Deep Dive

### Stage 1: Multilingual Preprocessing

**Module:** [`src/preprocess.py`](src/preprocess.py)

The preprocessing pipeline normalises noisy, multilingual text without losing discriminatory signal:

| Technique | What It Does | Example |
|-----------|-------------|---------|
| **NFKD Unicode decomposition** | Strips accents/diacritics → ASCII | `Lumière` → `lumiere` |
| **Legal suffix stripping** | Removes corporate designations (US/India/France) | `Bharat Pvt Ltd` → `bharat` |
| **Postal code extraction** | Country-aware ZIP/PIN/postal parsing | US: `95134`, India: `560001`, France: `75008` |
| **Street standardisation** | Maps abbreviations to full forms | `st` → `street`, `r` → `rue`, `bd` → `boulevard` |
| **Prefix token signatures** | 3/4-char prefixes for typo resilience | `Williams` → `pfx:wil`, `pfx:will` |

**Supported Legal Suffixes:**
- 🇺🇸 US: `Corporation`, `Corp`, `Inc`, `LLC`, `LLP`, `Ltd`, `Co`
- 🇮🇳 India: `Private Limited`, `Pvt Ltd`, `Public Limited`
- 🇫🇷 France: `SAS`, `SA`, `SARL`, `EURL`, `SNC`, `Société Anonyme`

### Stage 2: Country-Partitioned Blocking

**Module:** [`src/blocking.py`](src/blocking.py)

Reduces the comparison space by **99.999%** — from ~10¹³ possible pairs to ≤50 candidates per S1 entity.

**Architecture:**
1. **Hard country partitioning** — zero cross-border comparisons
2. **Multi-channel inverted index** with weighted signals:

   | Channel | Key Format | Weight | Signal Strength |
   |---------|-----------|--------|----------------|
   | Postal code | `post:{code}` | **+5** | Strongest — very precise |
   | Address token | `addr:{token}` | **+3** | City/street names — robust to transliteration |
   | Name token | `name:{token}` | **+2** | Core business name words |
   | Address number | `num:{number}` | **+2** | Door/building numbers |

3. **Max-DF filtering** — removes overly-common tokens (> 3% document frequency)
4. **Top-50 truncation** by weighted hit score

**Blocking Recall:** 99.48% of true matches captured in candidate pools.

### Stage 3: Pairwise Feature Engineering

**Module:** [`src/features.py`](src/features.py)

Extracts **26 features** across 4 channels for each candidate pair, powered by C-accelerated `rapidfuzz`:

<details>
<summary><b>Full Feature List (26 features)</b></summary>

**Name Similarity (10 features):**
| Feature | Description |
|---------|-------------|
| `jaro_winkler_sim` | Prefix-biased string similarity |
| `token_sort_ratio` | Order-insensitive token similarity |
| `token_set_ratio` | Subset-aware token similarity |
| `levenshtein_ratio` | Character-level edit distance similarity |
| `exact_name_match` | Binary exact match flag |
| `name_len_diff` | Absolute character length difference |
| `name_len_ratio` | Ratio of shorter to longer name |
| `name_token_jaccard` | Token-level Jaccard overlap |
| `name_first_word_jw` | First significant word similarity |
| `name_char_jaccard` | Character-level Jaccard overlap |

**Address Similarity (9 features):**
| Feature | Description |
|---------|-------------|
| `postal_code_match` | Tri-state: +1 (match), 0 (missing), -1 (conflict) |
| `postal_exact_binary` | Binary postal match flag |
| `address_number_jaccard` | Door/building number overlap |
| `address_number_conflict` | Binary conflicting numbers flag |
| `address_token_overlap` | Address word token Jaccard |
| `address_levenshtein_ratio` | Address string edit distance similarity |
| `address_jaro_winkler` | Address string prefix similarity |
| `address_has_missing` | Missing address indicator |
| `digit_overlap_ratio` | All-digit overlap across name+address |

**Cross-Field Signals (3 features):**
| Feature | Description |
|---------|-------------|
| `combined_jw` | Name+address concatenated similarity |
| `name_has_non_latin` | Non-Latin character detection (transliteration flag) |
| `address_number_count_diff` | Numerical count discrepancy |

**Metadata (4 features):**
| Feature | Description |
|---------|-------------|
| `is_source2` | Source 2 indicator |
| `is_source3` | Source 3 indicator |
| `blocking_rank` | Blocking engine rank position |
| `blocking_rank_inv` | Inverse blocking rank |

</details>

**Throughput:** ~28,700 pairs/second.

### Stage 4: LightGBM + BSCSG Decision Engine

**Module:** [`src/model.py`](src/model.py)

#### Model Configuration

```
LGBMClassifier(
    n_estimators=500, max_depth=7, num_leaves=63,
    learning_rate=0.05, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8
)
```

#### The Brutally Strict Confidence & Singleton Gate (BSCSG)

The BSCSG is the core precision-optimisation strategy, built around the mathematical reality of F₀.₅:

1. **Fine-Grained Threshold Calibration (τ\*):**
   Grid search from 0.30–0.98 with step=0.005, directly optimising Macro F₀.₅ on validation data.

2. **Smart Postal Veto:**
   Only vetoes when **both** postal codes are 5+ digits **and** differ. Avoids false vetoes from partial/missing codes.

3. **Smart Number Conflict Veto:**
   Only vetoes when **both** sides have 2+ address numbers with **zero** overlap.

4. **Walk-Away Policy:**
   If max candidate probability < τ\*, predict empty — earning a guaranteed **1.0** on true singletons.

---

## 📊 Evaluation & Metrics

**Module:** [`utils/metrics.py`](utils/metrics.py)

The exact competition metric implementation with special singleton handling:

| Scenario | True Matches | Prediction | F₀.₅ Score |
|----------|-------------|------------|-----------|
| Correct singleton | ∅ | ∅ | **1.0** ✅ |
| False positive on singleton | ∅ | {S2-001} | **0.0** ❌ |
| Perfect match | {S2-001} | {S2-001} | **1.0** ✅ |
| Missed match | {S2-001} | ∅ | **0.0** ❌ |
| Partial match | {S2-001, S2-002} | {S2-001} | 0.833 |

---

## 🧪 Testing

The project includes **32 unit tests** covering all pipeline components:

```bash
pytest tests/ -v
```

| Test Module | Coverage |
|------------|---------|
| `test_preprocess.py` | Unicode normalisation, suffix stripping, postal extraction, address tokenisation |
| `test_blocking.py` | Inverted index construction, candidate generation, country partitioning |
| `test_features.py` | All 26 feature computations, edge cases, missing data handling |
| `test_model.py` | Model training, threshold calibration, BSCSG veto logic |
| `test_metrics.py` | F₀.₅ computation, singleton rules, macro averaging |
| `test_validator.py` | Submission format compliance, subset rules, duplicate detection |

---

## 📓 Kaggle Deployment

The self-contained Kaggle notebook (`business_entity_resolution_kaggle.ipynb`) runs the complete pipeline end-to-end within Kaggle's constraints:

- **RAM:** Peak < 4 GB (within 30 GB limit)
- **Runtime:** Completes within 12-hour session limit
- **Disk:** ~20 GB Parquet (within 20 GB `/kaggle/working` limit)
- **Dependencies:** All pinned in `requirements.txt`

---

## ⚖️ Constraints & Compliance

| Constraint | Status |
|-----------|--------|
| No external data / geocoding / APIs | ✅ Strictly zero |
| Model ≤ 8B parameters | ✅ LightGBM (~500 trees) |
| Open-source license (MIT/Apache 2.0) | ✅ Fully compliant |
| Kaggle 30 GB RAM limit | ✅ Peak < 4 GB |
| Kaggle 12-hour session limit | ✅ Completes in minutes |
| `matched_entity_ids` ⊆ `candidate_entity_ids` | ✅ Enforced by validator |

---

## 👥 Team

**Team Zenith**

Built for the Amazon ML Challenge 2026 (September 25–27, 2026).

---

<p align="center">
  <i>Precision is not paranoia — it's the optimal strategy when F₀.₅ is the objective.</i>
</p>
