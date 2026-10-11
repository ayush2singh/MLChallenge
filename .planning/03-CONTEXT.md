# Phase 3 Context: Out-of-Core Blocking & Candidate Generation

**Phase Target**: Phase 3 (Out-of-Core Blocking & Candidate Generation)  
**Parent Milestone**: Business Entity Resolution Challenge  
**Status**: Decided & Ready for Planning (`/gsd-plan-phase 3`)

---

## 1. Executive Summary & Phase Goal

Phase 3 is responsible for bridging raw multi-source tabular data (Source 1, Source 2, Source 3) to a bounded, high-recall candidate set.

- **Primary Objective**: Reduce the $O(|S1| \times (|S2| + |S3|))$ Cartesian product space to a compact pool of at most **30 candidate entities per Source 1 entity**, achieving a **recall ceiling $\ge 98\%$** on ground-truth matches while strictly operating under $< 4\text{ GB}$ peak memory.
- **Output Artifact**: `output/candidate_pairs.tsv` formatted strictly according to competition requirements:
  ```tsv
  source1_entity_id\tcandidate_entity_ids
  S1-00001\tS2-00047,S2-00193,S3-00812
  S1-00002\tS3-00004
  S1-00003\t
  ```

---

## 2. Locked Implementation Decisions

Following structured architectural discovery with the user, the following design choices are finalized and locked for downstream planner and execution agents:

### Decision 1: Candidate Generation Architecture — Dual-Layer Ensemble
- **Mechanism**:
  - **Hard Country Partitioning**: Partition datasets strictly by `country` (open set handling US, India, France, and any unseen country labels dynamically). Candidates are only matched within the same country partition.
  - **Layer 1 (Token Inverted Index)**: Fast inverted index hashing on:
    1. Cleaned core business name tokens (after stopword and legal suffix removal).
    2. Numerical address signatures (postal/PIN codes, door/street numbers).
  - **Layer 2 (Sparse TF-IDF Char N-Gram Cosine Retrieval)**:
    - Character 2-4 n-grams with sublinear term-frequency weighting (`sublinear_tf=True`).
    - Cosine similarity computation between S1 and S2/S3 representation vectors within each country partition.
  - **Candidate Union & Rank**:
    - Union candidates from Layer 1 and Layer 2.
    - Rank by cosine similarity and keep the top candidates within the budget.
- **Rationale**: Single inverted index hashing is susceptible to spelling typos; pure TF-IDF matrix dot-products can miss exact landmark or PIN code links when names vary widely. Combining them ensures an upper-bound recall $\ge 98\%$.

### Decision 2: Candidate Pool Budget ($K$ per Source 1 Entity) — Fixed Top-30
- **Budget**: Exactly $\le 30$ candidates per Source 1 entity.
- **Rationale**:
  - Keeps pairwise feature extraction (Phase 4) extremely fast and lightweight.
  - Bounded memory footprint ($< 4\text{ GB}$ RAM) comfortably inside local 8GB development laptops and Kaggle's 30GB environment.
  - Generates compact `candidate_pairs.tsv` on 50–100GB datasets.
  - If blocking retrieves 0 candidates for an S1 entity (or all similarities fall below a noise floor $\tau_{\text{floor}} = 0.05$), the candidate list is intentionally left blank (`""`), representing a probable singleton.

### Decision 3: Text Normalization & Multilingual Handling — NFKD Normalization + Multilingual Legal Suffix Stripper
- **Mechanism**:
  - **Unicode Normalization**: Apply `unicodedata.normalize('NFKD', text)` to decompose and strip diacritics and accents (critical for French entities in test data: `é, è, ê, ç, ô, î, ï, à`).
  - **Legal Suffix Sanitization**: Multilingual suffix map covering:
    - **US**: `Corp`, `Corporation`, `Inc`, `Incorporated`, `LLC`, `L.L.C.`, `Co`, `Company`, `Ltd`.
    - **India**: `Pvt Ltd`, `Private Limited`, `Ltd`, `Limited`, `LLP`.
    - **France**: `SA`, `SAS`, `S.A.S.`, `SARL`, `S.A.R.L.`, `EURL`, `SNC`, `Societe Anonyme`.
  - **Address Tokenization**:
    - Extract postal codes (US 5-digit ZIP / ZIP+4, India 6-digit PIN, France 5-digit postal code).
    - Standardize common street abbreviations (`St -> Street`, `Rd -> Road`, `Ave -> Avenue`, `Blvd -> Boulevard`, `Rue -> Rue`).
    - Remove punctuation, lowercase, collapse whitespace.
- **Rationale**: Open-set evaluation includes France alongside US and India. Normalizing diacritics and international corporate forms prevents vocabulary mismatch across noisy data sources.

---

## 3. Module Boundaries & Responsibilities

| Module | Location | Responsibilities | Key Dependencies |
| :--- | :--- | :--- | :--- |
| **Preprocessor** | [preprocess.py](file:///c:/Users/AYUSH/Documents/MLChallenge/src/preprocess.py) | - Streaming TSV chunk reader<br>- Multilingual string cleaner & NFKD diacritic stripper<br>- Legal suffix removal and core token extraction<br>- Address signature & postal code extractor | `polars`, `re`, `unicodedata` |
| **Blocking Engine** | [blocking.py](file:///c:/Users/AYUSH/Documents/MLChallenge/src/blocking.py) | - Country-partitioned execution harness<br>- Layer 1 inverted index token generator<br>- Layer 2 sparse char n-gram TF-IDF retriever<br>- Candidate union, deduplication, and Top-30 truncation | `scipy.sparse`, `sklearn.feature_extraction.text`, `polars` / `numpy` |
| **Candidate Exporter** | [blocking.py](file:///c:/Users/AYUSH/Documents/MLChallenge/src/blocking.py) | - Serializes `output/candidate_pairs.tsv`<br>- Validates format using [utils/validate_submission.py](file:///c:/Users/AYUSH/Documents/MLChallenge/utils/validate_submission.py) | Python stdlib `csv` |

---

## 4. Quality Bar & Verification Plan

Downstream execution of Phase 3 must pass the following verification gates before marking Phase 3 complete:

1. **Unit Tests (`tests/test_preprocess.py`)**:
   - Verify NFKD stripping on French test strings (e.g. `Atelier Lumière Artisanat SAS` -> `atelier lumiere artisanat`).
   - Verify legal suffix stripping across US, India, and France variants.
   - Verify postal code and address token extraction.
2. **Unit Tests (`tests/test_blocking.py`)**:
   - Verify country partition isolation (Source 2/3 entities from US never match an S1 entity from France or India).
   - Verify Top-30 cap per S1 entity.
   - Verify candidate list format: comma-separated, no quotes, sorted/clean.
3. **Recall Ceiling Benchmark on Train Ground Truth**:
   - Evaluate candidate set against `dataset/train/train_ground_truth.tsv`.
   - Candidate recall ceiling must be $\ge 98\%$.
4. **Validation Harness Pass**:
   - Run `python utils/validate_submission.py --matching ... --candidate output/candidate_pairs.tsv --test-dir dataset/test`.
   - Must achieve exit code 0 (`PASS`).
