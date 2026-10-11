# Business Entity Resolution Pipeline

**Amazon ML Challenge 2026**  
**Task**: Scalable Multi-Source Business Entity Resolution under Macro $F_{0.5}$  
**Target Output**: `output/matching_results.tsv` and `output/candidate_pairs.tsv`

---

## 1. Overview & Architecture

This package provides a self-contained, high-performance Entity Resolution (ER) solution designed to link deduplicated reference businesses (**Source 1**) to noisy records (**Source 2** and **Source 3**) across open-set country partitions (US, India, and France).

The system consists of an end-to-end all-in-one pipeline (`src/main.py`) that performs:
1. **Multilingual Streaming Preprocessor**:
   - `unidecode` transliteration for non-Latin scripts (Hindi, Bengali, etc.).
   - Multilingual corporate legal suffix stripper and domain suffix removal.
2. **Country-Partitioned Dual-Layer Blocking Engine**:
   - Hard country partitioning guaranteeing zero cross-border Cartesian product explosion.
   - Numba-optimized index lookup.
3. **Feature Engineering & LightGBM Matching**:
   - C-accelerated RapidFuzz string metrics.
   - 37 handcrafted features including containment, substring matching, and sorted Jaro-Winkler.
   - Pre-scoring veto optimization that doubles inference speed.

---

## 2. Directory Structure

```text
business_entity_resolution/
├── src/
│   └── main.py               # End-to-end pipeline script
├── requirements.txt          # Pinned Python dependencies
└── README.md                 # Reproduction instructions (this document)
```

---

## 3. Environment Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.10, 3.11, 3.12)
- At least 4 GB of RAM (Pipeline optimized for < 4GB peak memory usage)

### Installation
```bash
# 1. (Optional) Create and activate virtual environment
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 4. How to Run the Pipeline

### Step 1: Place Data
Ensure training and test TSVs are placed in `dataset/` (relative to where you run the script, or modify the paths inside the script, the script automatically discovers paths):
```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

### Step 2: Execute End-to-End Pipeline
Run the all-in-one script:
```bash
python src/main.py
```

This single command:
1. Discovers the training and test dataset paths automatically.
2. Trains the LightGBM model on 100,000 samples and calibrates the threshold.
3. Streams predictions country-by-country onto the disk, producing `output/candidate_pairs.tsv` and `output/matching_results.tsv`.
4. Performs a final validation check to ensure format constraints are met.

---

## 5. License & Fair Play Compliance

- **Fair Play**: ZERO external APIs, geocoding lookups, or internet databases were used. All predictions are generated purely from the provided training data.
- **Model License**: LightGBM is open-source (MIT License). Total parameters $\ll 8\text{B}$.
