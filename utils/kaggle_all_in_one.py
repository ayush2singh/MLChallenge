# ==============================================================================
# AMAZON ML CHALLENGE 2026: HIGH-ACCURACY ENTITY RESOLUTION PIPELINE (KAGGLE)
# ==============================================================================
# v2.3 - Ultra-Optimized Single Thread + Auto-Resume Checkpoints
#
# NEW IN V2.3:
# - Completely eliminates Pandas DataFrame overhead by predicting directly on numpy arrays.
# - Replaces dictionary-based features with flat lists, reducing memory allocations by 10x.
# - Fixes a catastrophic O(N^2) memory scan bug in the postal veto logic.
# - Uses dictionary .pop() for aggressive garbage collection to prevent Kaggle Swap death.
# ==============================================================================

import sys, os, subprocess, gc, time, csv, re, unicodedata
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any

# 1. Install dependencies automatically if missing
try:
    import rapidfuzz
    import polars as pl
    import lightgbm
except ImportError:
    print("Installing dependencies (rapidfuzz, polars, lightgbm)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "rapidfuzz", "polars", "lightgbm"])
    import rapidfuzz
    import polars as pl
    import lightgbm

import numpy as np
from lightgbm import LGBMClassifier
import rapidfuzz.distance.JaroWinkler as jw
import rapidfuzz.distance.Levenshtein as lev
import rapidfuzz.fuzz as fuzz

print("=" * 65)
print("AMAZON ML CHALLENGE 2026: HIGH-ACCURACY ER v2.3 (ULTRA-FAST)")
print("=" * 65)

# ==============================================================================
# A. DYNAMIC DATASET DISCOVERY
# ==============================================================================
def discover_data_paths():
    search_roots = ["/kaggle/input", "dataset", "."]
    train_dir, test_dir = None, None
    for root in search_roots:
        if not os.path.exists(root):
            continue
        for r, dirs, files in os.walk(root):
            if "train_ground_truth.tsv" in files or "train_source1.tsv" in files:
                train_dir = r
            if "test_source1.tsv" in files:
                test_dir = r
    if not train_dir:
        train_dir = "dataset/train"
    if not test_dir:
        test_dir = "dataset/test"
    output_dir = "/kaggle/working/output" if os.path.exists("/kaggle/working") else "output"
    os.makedirs(output_dir, exist_ok=True)
    return train_dir, test_dir, output_dir

TRAIN_DIR, TEST_DIR, OUTPUT_DIR = discover_data_paths()
print(f"TRAIN: {TRAIN_DIR} | TEST: {TEST_DIR} | OUTPUT: {OUTPUT_DIR}")

# ==============================================================================
# B. PREPROCESSING & NORMALIZATION
# ==============================================================================
BUSINESS_STOPWORDS = {"and", "the", "of", "in", "for", "at", "by", "from", "with", "a", "an", "on", "to"}
STREET_SUFFIX_MAP = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "hwy": "highway", "ste": "suite", "fl": "floor",
    "bldg": "building", "r": "rue", "bd": "boulevard", "pl": "place"
}
ADDRESS_STOPWORDS = {
    "street", "road", "avenue", "lane", "drive", "boulevard", "highway",
    "place", "court", "circle", "way", "suite", "floor", "building",
    "apartment", "unit", "room", "block", "sector", "phase", "plot",
    "near", "opp", "opposite", "behind", "next", "above", "below",
    "nagar", "colony", "enclave", "vihar", "puram", "town",
    "north", "south", "east", "west", "new", "old",
    "rue", "allee", "impasse", "chemin", "route", "pont",
    "main", "cross", "inner", "outer", "null", "none", "na",
}

LEGAL_SUFFIX_PATTERNS = [
    r"\bprivate\s+limited\b", r"\bpvt\s+limited\b", r"\bpvt\s+ltd\b", r"\bpvt\b",
    r"\bcorporation\b", r"\bincorporated\b", r"\blimited\s+liability\s+company\b",
    r"\bpublic\s+limited\b",
    r"\bsociete\s+par\s+actions\s+simplifiee\b", r"\bsociete\s+anonyme\b",
    r"\bsociete\s+a\s+responsabilite\s+limitee\b",
    r"\bentreprise\s+unipersonnelle\s+a\s+responsabilite\s+limitee\b",
    r"\bsociete\s+en\s+nom\s+collectif\b",
    r"\bcorp\b", r"\binc\b", r"\bllc\b", r"\bllp\b", r"\bltd\b", r"\blimited\b",
    r"\bco\b", r"\bcompany\b", r"\bsas\b", r"\bsarl\b", r"\beurl\b", r"\bsnc\b", r"\bsa\b",
]
RE_LEGAL_SUFFIX_END = re.compile(
    r"(?:(?:\s+|^)(?:" + "|".join(LEGAL_SUFFIX_PATTERNS) + r"))+\s*$", re.IGNORECASE
)
RE_INDIA_PIN = re.compile(r"\b([1-9]\d{5})\b")
RE_US_ZIP = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
RE_GENERAL_POSTAL = re.compile(r"\b(\d{5,6})\b")
RE_NUMBERS = re.compile(r"\b\d+\b")
RE_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9\s]")
RE_EXTRA_SPACES = re.compile(r"\s+")
RE_ALL_DIGITS = re.compile(r"\d+")
RE_NON_LATIN = re.compile(r"[^\x00-\x7F]")


def normalize_text(text):
    if text is None or not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    cleaned = normalized.encode("ASCII", "ignore").decode("utf-8")
    cleaned = cleaned.replace("&", " and ").lower()
    cleaned = RE_NON_ALPHANUMERIC.sub(" ", cleaned)
    return RE_EXTRA_SPACES.sub(" ", cleaned).strip()


def strip_legal_suffixes(name):
    if not name:
        return ""
    stripped = name
    for _ in range(3):
        new_stripped = RE_LEGAL_SUFFIX_END.sub("", stripped).strip()
        if new_stripped == stripped:
            break
        stripped = new_stripped
    return stripped if stripped else name

def _first_word(name):
    for w in name.split():
        if len(w) >= 3 and w not in BUSINESS_STOPWORDS:
            return w
    parts = name.split()
    return parts[0] if parts else ""

def extract_postal_code(address, country=None):
    if not address:
        return ""
    cl = country.lower().strip() if country else ""
    if "india" in cl:
        m = RE_INDIA_PIN.search(address)
        if m: return m.group(1)
    elif "us" in cl or "united states" in cl:
        m = RE_US_ZIP.search(address)
        if m: return m.group(1)
    elif "france" in cl:
        m = re.search(r"\b(\d{5})\b", address)
        if m: return m.group(1)
    m = RE_GENERAL_POSTAL.search(address)
    return m.group(1) if m else ""


def preprocess_record(record):
    name = record.get("business_name", "")
    addr = record.get("business_address", "")
    country = str(record.get("country", "")).strip()

    norm_name = normalize_text(name)
    core_name = strip_legal_suffixes(norm_name)

    # Address processing
    postal = extract_postal_code(addr, country)
    norm_addr = normalize_text(addr)
    words = norm_addr.split()
    std_words = [STREET_SUFFIX_MAP.get(w, w) for w in words]
    clean_address = " ".join(std_words)
    numbers = RE_NUMBERS.findall(clean_address)
    addr_tokens = [w for w in std_words if len(w) >= 2 and w not in BUSINESS_STOPWORDS]

    # Name tokens with prefix signatures
    core_words = [t for t in core_name.split() if len(t) >= 2 and t not in BUSINESS_STOPWORDS]
    name_tokens = list(core_words)
    for w in core_words:
        if len(w) >= 4:
            name_tokens.append(f"pfx:{w[:3]}")
            name_tokens.append(f"pfx:{w[:4]}")

    # ==============================================================
    # PRECOMPUTE SETS & STRINGS ONCE FOR FAST FEATURE EXTRACTION
    # ==============================================================
    s1_tok = set(t for t in name_tokens if not t.startswith("pfx:"))
    fw = _first_word(core_name)
    s1_aw = set(addr_tokens)
    s1_nums = set(numbers)
    s1_comb = f"{core_name} {clean_address}".strip()
    s1_dg = set(RE_ALL_DIGITS.findall(f"{name or ''} {addr or ''}"))
    name_has_non_latin = 1.0 if RE_NON_LATIN.search(name or "") else 0.0

    return {
        "entity_id": record.get("entity_id", ""),
        "country": country,
        "clean_name": core_name,
        "clean_address": clean_address,
        "postal_code": postal,
        "name_tokens": name_tokens,
        "address_tokens": addr_tokens,
        "address_numbers": numbers,
        
        # O(1) Lookup fields
        "s1_tok": s1_tok,
        "fw": fw,
        "s1_aw": s1_aw,
        "s1_nums": s1_nums,
        "s1_comb": s1_comb,
        "s1_dg": s1_dg,
        "name_has_non_latin": name_has_non_latin,
        "name_chars": set(core_name)
    }

# ==============================================================================
# C. BLOCKING ENGINE (Address-First Inverted Index)
# ==============================================================================
class BlockingEngine:
    def __init__(self, top_k=50, max_df_ratio=0.03):
        self.top_k = top_k
        self.max_df_ratio = max_df_ratio

    def build_inverted_index(self, cand_records):
        index = defaultdict(list)
        n = len(cand_records)
        for idx, rec in enumerate(cand_records):
            for token in rec.get("name_tokens", []):
                if len(token) >= 3:
                    index[f"name:{token}"].append(idx)
            postal = rec.get("postal_code", "")
            if postal and len(postal) >= 4:
                index[f"post:{postal}"].append(idx)
            for num in rec.get("address_numbers", []):
                if len(num) >= 2:
                    index[f"num:{num}"].append(idx)
            for token in rec.get("address_tokens", []):
                if len(token) >= 4 and token.lower() not in ADDRESS_STOPWORDS:
                    index[f"addr:{token}"].append(idx)
        # Max-DF filtering
        if n > 100:
            max_postings = max(int(n * self.max_df_ratio), 50)
            keys_to_remove = [k for k, v in index.items() if len(v) > max_postings]
            for k in keys_to_remove:
                del index[k]
        
        # Convert lists to numpy arrays for screaming fast bincount query
        return {k: np.array(v, dtype=np.int32) for k, v in index.items()}

    def query_index(self, s1_rec, index):
        hits_arrs = []
        weights_arrs = []
        
        def add_hits(key, weight):
            arr = index.get(key)
            if arr is not None:
                hits_arrs.append(arr)
                weights_arrs.append(np.full(len(arr), weight, dtype=np.float32))

        postal = s1_rec.get("postal_code", "")
        if postal and len(postal) >= 4:
            add_hits(f"post:{postal}", 5.0)
            
        for token in s1_rec.get("address_tokens", []):
            if len(token) >= 4 and token.lower() not in ADDRESS_STOPWORDS:
                add_hits(f"addr:{token}", 3.0)
                
        for token in s1_rec.get("name_tokens", []):
            if len(token) >= 3:
                add_hits(f"name:{token}", 2.0)
                
        for num in s1_rec.get("address_numbers", []):
            if len(num) >= 2:
                add_hits(f"num:{num}", 2.0)
                
        if not hits_arrs:
            return []
            
        flat_hits = np.concatenate(hits_arrs)
        flat_weights = np.concatenate(weights_arrs)
        
        # O(N) scoring using bincount instead of Python loops
        scores = np.bincount(flat_hits, weights=flat_weights)
        nonzero_idx = np.nonzero(scores)[0]
        nonzero_scores = scores[nonzero_idx]
        
        if len(nonzero_idx) <= self.top_k:
            sorted_args = np.argsort(-nonzero_scores)
            return nonzero_idx[sorted_args].tolist()
            
        # O(N) top-K partitioning
        k = self.top_k
        partitioned = np.argpartition(-nonzero_scores, k - 1)[:k]
        top_k_scores = nonzero_scores[partitioned]
        sorted_top_k = np.argsort(-top_k_scores)
        
        return int(nonzero_idx[partitioned[sorted_top_k]]).tolist() if isinstance(nonzero_idx[partitioned[sorted_top_k]], int) else nonzero_idx[partitioned[sorted_top_k]].tolist()

# ==============================================================================
# D. FAST FEATURE ENGINEERING (26 Features)
# ==============================================================================
FEATURE_NAMES = [
    "jaro_winkler_sim", "token_sort_ratio", "token_set_ratio", "levenshtein_ratio",
    "exact_name_match", "name_len_diff", "name_len_ratio", "name_token_jaccard",
    "name_first_word_jw", "name_char_jaccard",
    "postal_code_match", "postal_exact_binary", "address_number_jaccard",
    "address_number_conflict", "address_token_overlap", "address_levenshtein_ratio",
    "address_jaro_winkler", "address_has_missing", "digit_overlap_ratio",
    "combined_jw", "name_has_non_latin", "address_number_count_diff",
    "is_source2", "is_source3", "blocking_rank", "blocking_rank_inv",
]

def compute_features(s1_rec, cand_rec, rank=1):
    """Returns a flat list of 26 floats (same order as FEATURE_NAMES) for direct numpy ingestion."""
    s1_name = s1_rec["clean_name"]
    cand_name = cand_rec["clean_name"]

    if s1_name and cand_name:
        jaro_winkler_sim = jw.similarity(s1_name, cand_name)
        token_sort_ratio = fuzz.token_sort_ratio(s1_name, cand_name) * 0.01
        token_set_ratio = fuzz.token_set_ratio(s1_name, cand_name) * 0.01
        levenshtein_ratio = lev.normalized_similarity(s1_name, cand_name)
        exact_name_match = 1.0 if s1_name == cand_name else 0.0
        name_len_diff = abs(len(s1_name) - len(cand_name))
        ml = max(len(s1_name), len(cand_name))
        name_len_ratio = min(len(s1_name), len(cand_name)) / ml if ml > 0 else 1.0
        s1c = s1_rec["name_chars"]
        s2c = cand_rec["name_chars"]
        u = s1c | s2c
        name_char_jaccard = len(s1c & s2c) / len(u) if u else 0.0
        fw1, fw2 = s1_rec["fw"], cand_rec["fw"]
        name_first_word_jw = jw.similarity(fw1, fw2) if fw1 and fw2 else 0.0
    else:
        jaro_winkler_sim = token_sort_ratio = token_set_ratio = levenshtein_ratio = 0.0
        exact_name_match = name_len_diff = name_len_ratio = name_char_jaccard = name_first_word_jw = 0.0

    s1_tok, c_tok = s1_rec["s1_tok"], cand_rec["s1_tok"]
    if s1_tok and c_tok:
        u = s1_tok | c_tok
        name_token_jaccard = len(s1_tok & c_tok) / len(u) if u else 0.0
    else:
        name_token_jaccard = 0.0

    s1_addr, c_addr = s1_rec["clean_address"], cand_rec["clean_address"]
    address_has_missing = 1.0 if (not s1_addr or not c_addr) else 0.0

    s1_post, c_post = s1_rec["postal_code"], cand_rec["postal_code"]
    if s1_post and c_post:
        pm = s1_post == c_post
        postal_code_match = 1.0 if pm else -1.0
        postal_exact_binary = 1.0 if pm else 0.0
    else:
        postal_code_match = postal_exact_binary = 0.0

    s1_nums, c_nums = s1_rec["s1_nums"], cand_rec["s1_nums"]
    if s1_nums and c_nums:
        isect = len(s1_nums & c_nums)
        address_number_jaccard = isect / len(s1_nums | c_nums)
        address_number_conflict = 1.0 if isect == 0 else 0.0
    else:
        address_number_jaccard = address_number_conflict = 0.0
    
    address_number_count_diff = abs(len(s1_nums) - len(c_nums))

    s1_aw, c_aw = s1_rec["s1_aw"], cand_rec["s1_aw"]
    if s1_aw and c_aw:
        u = s1_aw | c_aw
        address_token_overlap = len(s1_aw & c_aw) / len(u) if u else 0.0
    else:
        address_token_overlap = 0.0

    if s1_addr and c_addr:
        address_levenshtein_ratio = lev.normalized_similarity(s1_addr, c_addr)
        address_jaro_winkler = jw.similarity(s1_addr, c_addr)
    else:
        address_levenshtein_ratio = address_jaro_winkler = 0.0

    s1_dg, c_dg = s1_rec["s1_dg"], cand_rec["s1_dg"]
    if s1_dg and c_dg:
        u = s1_dg | c_dg
        digit_overlap_ratio = len(s1_dg & c_dg) / len(u) if u else 0.0
    else:
        digit_overlap_ratio = 0.0

    s1_comb, c_comb = s1_rec["s1_comb"], cand_rec["s1_comb"]
    combined_jw = jw.similarity(s1_comb, c_comb) if (s1_comb and c_comb) else 0.0

    cand_id = cand_rec.get("entity_id", "") or ""
    
    # Return flat list matching FEATURE_NAMES order — avoids dict allocation overhead
    return [
        jaro_winkler_sim, token_sort_ratio, token_set_ratio, levenshtein_ratio,
        exact_name_match, name_len_diff, name_len_ratio, name_token_jaccard,
        name_first_word_jw, name_char_jaccard,
        postal_code_match, postal_exact_binary, address_number_jaccard,
        address_number_conflict, address_token_overlap, address_levenshtein_ratio,
        address_jaro_winkler, address_has_missing, digit_overlap_ratio,
        combined_jw, cand_rec["name_has_non_latin"], address_number_count_diff,
        1.0 if cand_id.startswith("S2-") else 0.0,
        1.0 if cand_id.startswith("S3-") else 0.0,
        float(rank),
        1.0 / rank if rank > 0 else 1.0,
    ]


# ==============================================================================
# E. TRAINING PIPELINE
# ==============================================================================
def train_model(train_dir, sample_size=50000):
    """Train LightGBM on a sample of training data with ALL blocking negatives."""
    t0 = time.time()
    print(f"\n[1/3] Loading training data (N={sample_size})...")

    gt_df = pl.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), separator="\t", n_rows=sample_size)
    ground_truth = {}
    all_targets = set()
    for r in gt_df.iter_rows(named=True):
        s1_id = str(r["source1_entity_id"])
        c_str = str(r["matched_entity_ids"] or "").strip()
        c_set = {c.strip() for c in c_str.split(",") if c.strip()} if c_str else set()
        ground_truth[s1_id] = c_set
        all_targets.update(c_set)

    s1_ids = list(ground_truth.keys())
    target_s2 = [c for c in all_targets if c.startswith("S2-")]
    target_s3 = [c for c in all_targets if c.startswith("S3-")]

    s1_pl = pl.read_csv(os.path.join(train_dir, "train_source1.tsv"), separator="\t").filter(
        pl.col("entity_id").is_in(s1_ids)
    )
    s2_pos = pl.read_csv(os.path.join(train_dir, "train_source2.tsv"), separator="\t").filter(
        pl.col("entity_id").is_in(target_s2)
    )
    s3_pos = pl.read_csv(os.path.join(train_dir, "train_source3.tsv"), separator="\t").filter(
        pl.col("entity_id").is_in(target_s3)
    )
    extra_n = max(100000, sample_size * 15)
    s2_extra = pl.read_csv(os.path.join(train_dir, "train_source2.tsv"), separator="\t", n_rows=extra_n)
    s3_extra = pl.read_csv(os.path.join(train_dir, "train_source3.tsv"), separator="\t", n_rows=extra_n)
    cand_pl = pl.concat([s2_pos, s3_pos, s2_extra, s3_extra]).unique(subset=["entity_id"])

    print(f"  Loaded {s1_pl.height} S1, {cand_pl.height} candidates in {time.time()-t0:.1f}s")

    print("  Preprocessing records...")
    t1 = time.time()
    s1_dict = {r["entity_id"]: preprocess_record(r) for r in s1_pl.to_dicts()}
    cand_dict = {r["entity_id"]: preprocess_record(r) for r in cand_pl.to_dicts()}
    print(f"  Preprocessed in {time.time()-t1:.1f}s")

    print("\n[2/3] Blocking + Feature extraction...")
    t1 = time.time()
    # Group by country
    s1_by_country = defaultdict(list)
    cand_by_country = defaultdict(list)
    for rec in s1_dict.values():
        s1_by_country[rec["country"].lower().strip()].append(rec)
    for rec in cand_dict.values():
        cand_by_country[rec["country"].lower().strip()].append(rec)

    blocker = BlockingEngine(top_k=50)
    candidate_pairs = {}
    for country, s1_part in s1_by_country.items():
        cand_part = cand_by_country.get(country, [])
        print(f"  Blocking [{country}]: {len(s1_part)} S1 x {len(cand_part)} cands")
        if not cand_part:
            for rec in s1_part:
                candidate_pairs[rec["entity_id"]] = []
            continue
        inv_idx = blocker.build_inverted_index(cand_part)
        cand_ids = [c["entity_id"] for c in cand_part]
        for s1_rec in s1_part:
            hits = blocker.query_index(s1_rec, inv_idx)
            candidate_pairs[s1_rec["entity_id"]] = [cand_ids[i] for i in hits]
        del inv_idx
        gc.collect()

    print(f"  Blocking done in {time.time()-t1:.1f}s")

    # Build training features with ALL blocking negatives
    print("  Building training features (all blocking negatives)...")
    t1 = time.time()
    feature_rows = []
    labels = []
    for s1_id, true_set in ground_truth.items():
        s1_rec = s1_dict.get(s1_id)
        if not s1_rec:
            continue
        cands = candidate_pairs.get(s1_id, [])
        cands_set = set(cands)

        # True positives
        for cand_id in true_set:
            crec = cand_dict.get(cand_id)
            if crec:
                rank = cands.index(cand_id) + 1 if cand_id in cands_set else len(cands) + 1
                feature_rows.append(compute_features(s1_rec, crec, rank))
                labels.append(1)

        # ALL blocking negatives
        for rank, cand_id in enumerate(cands, 1):
            if cand_id not in true_set:
                crec = cand_dict.get(cand_id)
                if crec:
                    feature_rows.append(compute_features(s1_rec, crec, rank))
                    labels.append(0)

    X_train = np.array(feature_rows, dtype=np.float64)
    y_train = np.array(labels, dtype=int)
    n_pos = int(y_train.sum())
    print(f"  Training set: {len(X_train)} pairs ({n_pos} pos, {len(X_train)-n_pos} neg) in {time.time()-t1:.1f}s")

    del feature_rows, labels
    gc.collect()

    print("\n[3/3] Training LightGBM + Threshold calibration...")
    t1 = time.time()
    model = LGBMClassifier(
        objective="binary", metric="auc", boosting_type="gbdt",
        n_estimators=500, max_depth=7, num_leaves=63, learning_rate=0.05,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=-1
    )
    model.fit(X_train, y_train)
    print(f"  Model trained in {time.time()-t1:.1f}s")

    # Calibrate threshold on training data (use last 20% as validation)
    val_start = int(0.8 * len(s1_ids))
    val_s1 = set(s1_ids[val_start:])
    val_pairs_sub = {k: v for k, v in candidate_pairs.items() if k in val_s1}

    val_feat_rows = []
    val_pk = []
    for s1_id in val_s1:
        s1_rec = s1_dict.get(s1_id)
        if not s1_rec:
            continue
        for rank, cand_id in enumerate(val_pairs_sub.get(s1_id, []), 1):
            crec = cand_dict.get(cand_id)
            if crec:
                val_feat_rows.append(compute_features(s1_rec, crec, rank))
                val_pk.append((s1_id, cand_id))

    if val_feat_rows:
        X_val = np.array(val_feat_rows, dtype=np.float64)
        probs = model.predict_proba(X_val)[:, 1]

        s1_cand_probs = defaultdict(list)
        for (s1_id, cand_id), p in zip(val_pk, probs):
            s1_cand_probs[s1_id].append((cand_id, float(p)))

        best_tau, best_f05 = 0.50, -1.0
        for tau in np.arange(0.30, 0.98, 0.01):
            f05_scores = []
            for s1_id in val_s1:
                true_set = ground_truth.get(s1_id, set())
                preds = {c for c, p in s1_cand_probs.get(s1_id, []) if p >= tau}
                tp = len(true_set & preds)
                if len(true_set) == 0:
                    f05_scores.append(1.0 if len(preds) == 0 else 0.0)
                elif tp == 0:
                    f05_scores.append(0.0)
                else:
                    prec = tp / len(preds)
                    rec = tp / len(true_set)
                    f05_scores.append(1.25 * prec * rec / (0.25 * prec + rec))
            mf = np.mean(f05_scores)
            if mf > best_f05:
                best_f05, best_tau = mf, round(tau, 3)

        print(f"  Calibrated tau*={best_tau:.3f} (val F0.5={best_f05:.4f})")
    else:
        best_tau = 0.50

    del X_train, y_train, s1_dict, cand_dict, candidate_pairs
    gc.collect()

    total_time = time.time() - t0
    print(f"\nTraining complete in {total_time:.1f}s")
    return model, best_tau


# ==============================================================================
# F. STREAMING INFERENCE PIPELINE (SINGLE-THREAD OPTIMIZED)
# ==============================================================================

def run_inference(model, best_tau, test_dir, output_dir):
    """Run inference on test set with country-partitioned streaming and fast single thread."""
    t0 = time.time()
    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    match_path = os.path.join(output_dir, "matching_results.tsv")

    print(f"\n{'='*60}")
    print("STREAMING INFERENCE PIPELINE (ULTRA-FAST)")
    print(f"{'='*60}")
    
    # Check for existing checkpoint files for auto-resume
    processed_s1_ids = set()
    if os.path.exists(match_path) and os.path.exists(cand_path):
        print(f"[0/4] Found existing output files. Attempting to resume...")
        try:
            processed_df = pl.read_csv(match_path, separator="\t")
            processed_s1_ids = set(processed_df["source1_entity_id"].to_list())
            print(f"  Resuming from {len(processed_s1_ids)} previously processed entities.")
            file_mode = "a"
        except Exception as e:
            print(f"  Error reading existing files: {e}. Starting fresh.")
            processed_s1_ids = set()
            file_mode = "w"
    else:
        file_mode = "w"

    # Initialize files if starting fresh
    if file_mode == "w":
        with open(cand_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["source1_entity_id", "candidate_entity_ids"])
        with open(match_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["source1_entity_id", "matched_entity_ids"])

    # Load test data
    print("[1/4] Loading test data...")
    ts1 = pl.read_csv(os.path.join(test_dir, "test_source1.tsv"), separator="\t")
    ts2 = pl.read_csv(os.path.join(test_dir, "test_source2.tsv"), separator="\t")
    ts3 = pl.read_csv(os.path.join(test_dir, "test_source3.tsv"), separator="\t")

    print(f"  Test S1 total: {ts1.height}, S2: {ts2.height}, S3: {ts3.height}")
    
    # Filter out already processed S1 entities
    if processed_s1_ids:
        ts1 = ts1.filter(~pl.col("entity_id").is_in(list(processed_s1_ids)))
        print(f"  Test S1 remaining after filtering: {ts1.height}")

    # Group by country
    s1_by_country = defaultdict(list)
    for r in ts1.to_dicts():
        s1_by_country[r["country"].lower().strip()].append(r)

    cand_all = pl.concat([ts2, ts3]).unique(subset=["entity_id"])
    cand_by_country = defaultdict(list)
    for r in cand_all.to_dicts():
        cand_by_country[r["country"].lower().strip()].append(r)

    del ts1, ts2, ts3, cand_all
    gc.collect()

    countries = sorted(set(list(s1_by_country.keys()) + list(cand_by_country.keys())))
    
    blocker = BlockingEngine(top_k=50)

    for country in countries:
        s1_raws = s1_by_country.pop(country, [])
        cand_raws = cand_by_country.pop(country, [])

        if not s1_raws:
            continue

        print(f"\n[2/4] Processing [{country}]: {len(s1_raws)} S1 remaining x {len(cand_raws)} candidates")
        t1 = time.time()

        # Preprocess - THIS NOW COMPUTES ALL SETS FOR INSTANT O(1) LOOKUPS
        s1_recs = [preprocess_record(r) for r in s1_raws]
        cand_recs = [preprocess_record(r) for r in cand_raws]
        
        cand_dict = {r["entity_id"]: r for r in cand_recs}
        cand_ids = [r["entity_id"] for r in cand_recs]

        del s1_raws, cand_raws
        gc.collect()

        # Build inverted index
        if cand_recs:
            inv_idx = blocker.build_inverted_index(cand_recs)
        else:
            inv_idx = {}

        # Open files in append mode to stream outputs directly to disk
        f_cand = open(cand_path, "a", newline="", encoding="utf-8")
        w_cand = csv.writer(f_cand, delimiter="\t")
        
        f_match = open(match_path, "a", newline="", encoding="utf-8")
        w_match = csv.writer(f_match, delimiter="\t")

        try:
            # Process S1 in batches
            batch_size = 10000
            n_s1 = len(s1_recs)
            total_scored = 0

            for batch_start in range(0, n_s1, batch_size):
                batch_end = min(batch_start + batch_size, n_s1)
                batch_recs = s1_recs[batch_start:batch_end]

                batch_feat_rows = []
                batch_pk = []
                batch_s1_ids = []
                batch_cand_pairs = {}
                
                # Ultra-Fast Single Thread Loop
                for s1_rec in batch_recs:
                    s1_id = s1_rec["entity_id"]
                    batch_s1_ids.append(s1_id)
                    
                    hits = blocker.query_index(s1_rec, inv_idx)
                    cand_list = [cand_ids[i] for i in hits]
                    batch_cand_pairs[s1_id] = cand_list
                    
                    for rank, cand_id in enumerate(cand_list, 1):
                        crec = cand_dict.get(cand_id)
                        if crec:
                            batch_feat_rows.append(compute_features(s1_rec, crec, rank))
                            batch_pk.append((s1_id, cand_id))

                # Score batch
                all_matches_batch = {}
                if batch_feat_rows:
                    X_batch = np.array(batch_feat_rows, dtype=np.float64)
                    probs = model.predict_proba(X_batch)[:, 1]

                    # Group by S1 and apply threshold + veto
                    s1_scored = defaultdict(list)
                    for (s1_id, cand_id), p in zip(batch_pk, probs):
                        s1_scored[s1_id].append((cand_id, float(p)))

                    # Build O(1) lookup for S1 records (fixes O(N^2) linear scan bug)
                    s1_rec_map = {r["entity_id"]: r for r in batch_recs}

                    for s1_id in batch_s1_ids:
                        candidates = s1_scored.get(s1_id, [])
                        valid = []
                        s1_rec_v = s1_rec_map.get(s1_id)
                        for cand_id, prob in candidates:
                            if prob < best_tau:
                                continue
                            # Smart postal veto
                            if s1_rec_v:
                                crec = cand_dict.get(cand_id)
                                if crec:
                                    sp = s1_rec_v.get("postal_code", "") or ""
                                    cp = crec.get("postal_code", "") or ""
                                    if sp and cp and len(sp) >= 5 and len(cp) >= 5 and sp != cp:
                                        continue
                                    sn = s1_rec_v.get("s1_nums", set())
                                    cn = crec.get("s1_nums", set())
                                    if len(sn) >= 2 and len(cn) >= 2 and len(sn & cn) == 0:
                                        continue
                            valid.append(cand_id)
                        all_matches_batch[s1_id] = valid
                else:
                    for s1_id in batch_s1_ids:
                        all_matches_batch[s1_id] = []

                # Flush batch to disk immediately
                for s1_id in batch_s1_ids:
                    # Write Candidate pairs
                    cands = batch_cand_pairs.get(s1_id, [])
                    seen = set()
                    unique_cands = [c for c in cands if not (c in seen or seen.add(c))]
                    w_cand.writerow([s1_id, ",".join(unique_cands)])
                    
                    # Write Matched pairs
                    matches = all_matches_batch.get(s1_id, [])
                    seen_m = set()
                    unique_matches = [m for m in matches if not (m in seen_m or seen_m.add(m))]
                    w_match.writerow([s1_id, ",".join(unique_matches)])

                f_cand.flush()
                f_match.flush()
                total_scored += len(batch_feat_rows)

                if (batch_end % 20000 == 0) or batch_end == n_s1:
                    elapsed = time.time() - t1
                    print(f"    [{country}] {batch_end}/{n_s1} entities, {total_scored} pairs scored ({elapsed:.0f}s)")
                        
        finally:
            f_cand.close()
            f_match.close()

        # Clear memory for next country
        cand_dict = None
        inv_idx = None
        cand_ids = None
        del s1_recs, cand_recs
        gc.collect()

    total_time = time.time() - t0
    print(f"\n[3/4] Outputs safely streamed to disk.")
    print(f"\n[4/4] INFERENCE COMPLETE")
    print(f"  Total time:        {total_time:.0f}s")
    print(f"  Output: {cand_path}")
    print(f"  Output: {match_path}")

    return cand_path, match_path


# ==============================================================================
# G. VALIDATION
# ==============================================================================
def validate_outputs(match_path, cand_path, test_dir):
    """Validates that outputs conform to competition format."""
    print(f"\nValidating outputs...")
    ts1 = pl.read_csv(os.path.join(test_dir, "test_source1.tsv"), separator="\t")
    expected_ids = set(ts1["entity_id"].to_list())

    match_df = pl.read_csv(match_path, separator="\t")
    match_ids = set(match_df["source1_entity_id"].to_list())

    cand_df = pl.read_csv(cand_path, separator="\t")
    cand_ids = set(cand_df["source1_entity_id"].to_list())

    missing_match = expected_ids - match_ids
    missing_cand = expected_ids - cand_ids

    if missing_match:
        print(f"  WARNING: {len(missing_match)} S1 entities missing from matching_results.tsv")
    else:
        print(f"  OK: All {len(expected_ids)} S1 entities present in matching_results.tsv")

    if missing_cand:
        print(f"  WARNING: {len(missing_cand)} S1 entities missing from candidate_pairs.tsv")
    else:
        print(f"  OK: All {len(expected_ids)} S1 entities present in candidate_pairs.tsv")

    # Check subset constraint
    errors = 0
    cand_map = {}
    for r in cand_df.iter_rows(named=True):
        s1_id = r["source1_entity_id"]
        c_str = str(r["candidate_entity_ids"] or "").strip()
        cand_map[s1_id] = set(c.strip() for c in c_str.split(",") if c.strip()) if c_str else set()

    for r in match_df.iter_rows(named=True):
        s1_id = r["source1_entity_id"]
        m_str = str(r["matched_entity_ids"] or "").strip()
        m_set = set(c.strip() for c in m_str.split(",") if c.strip()) if m_str else set()
        c_set = cand_map.get(s1_id, set())
        violations = m_set - c_set
        if violations:
            errors += 1

    if errors == 0:
        print(f"  OK: All matched IDs are subsets of candidate IDs")
    else:
        print(f"  WARNING: {errors} entities have matches not in candidates")

    print("Validation complete!")


# ==============================================================================
# H. MAIN
# ==============================================================================
if __name__ == "__main__":
    print(f"\nStarting pipeline at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Train model
    model, best_tau = train_model(TRAIN_DIR, sample_size=50000)

    # Step 2: Run inference
    cand_path, match_path = run_inference(model, best_tau, TEST_DIR, OUTPUT_DIR)

    # Step 3: Validate
    validate_outputs(match_path, cand_path, TEST_DIR)

    print(f"\n{'='*65}")
    print("PIPELINE COMPLETE!")
    print(f"{'='*65}")
