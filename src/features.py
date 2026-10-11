"""
Pairwise Feature Engineering Pipeline for Business Entity Resolution (v2).

Extracts 26 features across 4 channels:
1. Name Similarity (10 features): JW, Levenshtein, token-sort/set ratios, Jaccard,
   first-word similarity, character overlap, prefix match, length metrics, exact match.
2. Address Similarity (9 features): JW, Levenshtein, postal match, number overlap,
   token overlap, city signals, digit overlap, and missing-address indicators.
3. Cross-Field Signals (3 features): Combined text similarity, non-Latin name detection,
   address number count difference.
4. Metadata (4 features): Source indicators and blocking rank.

Key improvements over v1:
- address_jaro_winkler: Direct address string similarity
- name_first_word_jw: Isolated first-word comparison (robust to suffix noise)
- name_char_jaccard: Character-level overlap for typo/corruption resilience
- digit_overlap_ratio: All-digit overlap across name+address (very stable signal)
- combined_jw: Merged name+address similarity (global signal)
- name_has_non_latin: Detects transliterated names where name-based features fail
"""

from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import pandas as pd
import rapidfuzz.distance.JaroWinkler as jw
import rapidfuzz.distance.Levenshtein as lev
import rapidfuzz.fuzz as fuzz
import re

from src.preprocess import BUSINESS_STOPWORDS


FEATURE_NAMES: List[str] = [
    # Name similarities (10)
    "jaro_winkler_sim",
    "token_sort_ratio",
    "token_set_ratio",
    "levenshtein_ratio",
    "exact_name_match",
    "name_len_diff",
    "name_len_ratio",
    "name_token_jaccard",
    "name_first_word_jw",
    "name_char_jaccard",
    # Address similarities (9)
    "postal_code_match",
    "postal_exact_binary",
    "address_number_jaccard",
    "address_number_conflict",
    "address_token_overlap",
    "address_levenshtein_ratio",
    "address_jaro_winkler",
    "address_has_missing",
    "digit_overlap_ratio",
    # Cross-field signals (3)
    "combined_jw",
    "name_has_non_latin",
    "address_number_count_diff",
    # Metadata & source indicators (4)
    "is_source2",
    "is_source3",
    "blocking_rank",
    "blocking_rank_inv",
]

RE_ALL_DIGITS = re.compile(r"\d+")
RE_NON_LATIN = re.compile(r"[^\x00-\x7F]")


def _extract_all_digits(text: str) -> Set[str]:
    """Extracts all digit sequences from text."""
    return set(RE_ALL_DIGITS.findall(text)) if text else set()


def _char_jaccard(s1: str, s2: str) -> float:
    """Character-level Jaccard similarity between two strings."""
    if not s1 or not s2:
        return 0.0
    set1 = set(s1)
    set2 = set(s2)
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union > 0 else 0.0


def _first_significant_word(name: str) -> str:
    """Returns the first word that is >= 3 chars and not a stopword."""
    for word in name.split():
        if len(word) >= 3 and word not in BUSINESS_STOPWORDS:
            return word
    return name.split()[0] if name.split() else ""


def compute_pair_features(
    s1_rec: Dict[str, Any],
    cand_rec: Dict[str, Any],
    rank: int = 1
) -> Dict[str, float]:
    """
    Computes 26 pairwise similarity and discrepancy features between a Source 1 record
    and a candidate (Source 2 or Source 3) record.

    Args:
        s1_rec: Preprocessed Source 1 record dict
        cand_rec: Preprocessed Candidate record dict
        rank: Rank position of the candidate from the blocking stage (1-indexed)

    Returns:
        Dict mapping feature names to numerical values
    """
    # ----------------------------------------------------
    # 1. Name Features (10)
    # ----------------------------------------------------
    s1_name = s1_rec.get("clean_name", "") or ""
    cand_name = cand_rec.get("clean_name", "") or ""

    if s1_name and cand_name:
        jaro_winkler_sim = float(jw.similarity(s1_name, cand_name))
        token_sort_ratio = float(fuzz.token_sort_ratio(s1_name, cand_name)) / 100.0
        token_set_ratio = float(fuzz.token_set_ratio(s1_name, cand_name)) / 100.0
        levenshtein_ratio = float(lev.normalized_similarity(s1_name, cand_name))
        exact_name_match = 1.0 if s1_name == cand_name else 0.0
        name_len_diff = float(abs(len(s1_name) - len(cand_name)))
        max_len = max(len(s1_name), len(cand_name))
        name_len_ratio = float(min(len(s1_name), len(cand_name))) / float(max_len) if max_len > 0 else 1.0
        name_char_jaccard = _char_jaccard(s1_name, cand_name)

        # First significant word similarity
        s1_first = _first_significant_word(s1_name)
        cand_first = _first_significant_word(cand_name)
        name_first_word_jw = float(jw.similarity(s1_first, cand_first)) if s1_first and cand_first else 0.0
    else:
        jaro_winkler_sim = 0.0
        token_sort_ratio = 0.0
        token_set_ratio = 0.0
        levenshtein_ratio = 0.0
        exact_name_match = 0.0
        name_len_diff = float(abs(len(s1_name) - len(cand_name)))
        name_len_ratio = 0.0
        name_char_jaccard = 0.0
        name_first_word_jw = 0.0

    # Token Jaccard (exclude prefix tokens)
    s1_tokens = set(t for t in s1_rec.get("name_tokens", []) if not t.startswith("pfx:"))
    cand_tokens = set(t for t in cand_rec.get("name_tokens", []) if not t.startswith("pfx:"))

    if s1_tokens and cand_tokens:
        inter = len(s1_tokens & cand_tokens)
        union = len(s1_tokens | cand_tokens)
        name_token_jaccard = float(inter) / float(union) if union > 0 else 0.0
    else:
        name_token_jaccard = 0.0

    # ----------------------------------------------------
    # 2. Address Features (9)
    # ----------------------------------------------------
    s1_addr = s1_rec.get("clean_address", "") or ""
    cand_addr = cand_rec.get("clean_address", "") or ""
    address_has_missing = 1.0 if (not s1_addr or not cand_addr) else 0.0

    # Postal / PIN code matching:
    # postal_code_match: +1.0 = both exist and match, 0.0 = either missing, -1.0 = conflict
    # postal_exact_binary: 1.0 = both exist and match, 0.0 = otherwise
    s1_post = s1_rec.get("postal_code", "") or ""
    cand_post = cand_rec.get("postal_code", "") or ""

    if s1_post and cand_post:
        postal_code_match = 1.0 if s1_post == cand_post else -1.0
        postal_exact_binary = 1.0 if s1_post == cand_post else 0.0
    else:
        postal_code_match = 0.0
        postal_exact_binary = 0.0

    # Address numbers (door/building numbers)
    s1_nums = set(s1_rec.get("address_numbers", []))
    cand_nums = set(cand_rec.get("address_numbers", []))

    if s1_nums and cand_nums:
        num_inter = len(s1_nums & cand_nums)
        num_union = len(s1_nums | cand_nums)
        address_number_jaccard = float(num_inter) / float(num_union) if num_union > 0 else 0.0
        address_number_conflict = 1.0 if num_inter == 0 else 0.0
    else:
        address_number_jaccard = 0.0
        address_number_conflict = 0.0

    # Address word token overlap
    s1_addr_words = set(w for w in s1_addr.split() if len(w) >= 2 and w not in BUSINESS_STOPWORDS)
    cand_addr_words = set(w for w in cand_addr.split() if len(w) >= 2 and w not in BUSINESS_STOPWORDS)

    if s1_addr_words and cand_addr_words:
        addr_inter = len(s1_addr_words & cand_addr_words)
        addr_union = len(s1_addr_words | cand_addr_words)
        address_token_overlap = float(addr_inter) / float(addr_union) if addr_union > 0 else 0.0
    else:
        address_token_overlap = 0.0

    # Address string similarities
    if s1_addr and cand_addr:
        address_levenshtein_ratio = float(lev.normalized_similarity(s1_addr, cand_addr))
        address_jaro_winkler = float(jw.similarity(s1_addr, cand_addr))
    else:
        address_levenshtein_ratio = 0.0
        address_jaro_winkler = 0.0

    # Digit overlap: all digits found anywhere in the record (name + address combined)
    s1_all_text = f"{s1_rec.get('raw_name', '')} {s1_rec.get('clean_address', '')}"
    cand_all_text = f"{cand_rec.get('raw_name', '')} {cand_rec.get('clean_address', '')}"
    s1_digits = _extract_all_digits(s1_all_text)
    cand_digits = _extract_all_digits(cand_all_text)

    if s1_digits and cand_digits:
        digit_inter = len(s1_digits & cand_digits)
        digit_union = len(s1_digits | cand_digits)
        digit_overlap_ratio = float(digit_inter) / float(digit_union) if digit_union > 0 else 0.0
    else:
        digit_overlap_ratio = 0.0

    # Address number count difference
    address_number_count_diff = float(abs(len(s1_nums) - len(cand_nums)))

    # ----------------------------------------------------
    # 3. Cross-Field Signals (3)
    # ----------------------------------------------------
    # Combined name+address JW similarity
    s1_combined = f"{s1_name} {s1_addr}".strip()
    cand_combined = f"{cand_name} {cand_addr}".strip()
    if s1_combined and cand_combined:
        combined_jw = float(jw.similarity(s1_combined, cand_combined))
    else:
        combined_jw = 0.0

    # Non-Latin character detection in candidate name (transliteration signal)
    cand_raw_name = cand_rec.get("raw_name", "") or ""
    name_has_non_latin = 1.0 if RE_NON_LATIN.search(cand_raw_name) else 0.0

    # ----------------------------------------------------
    # 4. Source & Rank Indicators (4)
    # ----------------------------------------------------
    cand_id = cand_rec.get("entity_id", "") or ""
    is_source2 = 1.0 if cand_id.startswith("S2-") else 0.0
    is_source3 = 1.0 if cand_id.startswith("S3-") else 0.0
    blocking_rank = float(rank)
    blocking_rank_inv = 1.0 / float(rank) if rank > 0 else 1.0

    return {
        "jaro_winkler_sim": jaro_winkler_sim,
        "token_sort_ratio": token_sort_ratio,
        "token_set_ratio": token_set_ratio,
        "levenshtein_ratio": levenshtein_ratio,
        "exact_name_match": exact_name_match,
        "name_len_diff": name_len_diff,
        "name_len_ratio": name_len_ratio,
        "name_token_jaccard": name_token_jaccard,
        "name_first_word_jw": name_first_word_jw,
        "name_char_jaccard": name_char_jaccard,
        "postal_code_match": postal_code_match,
        "postal_exact_binary": postal_exact_binary,
        "address_number_jaccard": address_number_jaccard,
        "address_number_conflict": address_number_conflict,
        "address_token_overlap": address_token_overlap,
        "address_levenshtein_ratio": address_levenshtein_ratio,
        "address_jaro_winkler": address_jaro_winkler,
        "address_has_missing": address_has_missing,
        "digit_overlap_ratio": digit_overlap_ratio,
        "combined_jw": combined_jw,
        "name_has_non_latin": name_has_non_latin,
        "address_number_count_diff": address_number_count_diff,
        "is_source2": is_source2,
        "is_source3": is_source3,
        "blocking_rank": blocking_rank,
        "blocking_rank_inv": blocking_rank_inv,
    }


def extract_features_for_pairs(
    s1_dict: Dict[str, Dict[str, Any]],
    cand_dict: Dict[str, Dict[str, Any]],
    candidate_pairs: Dict[str, List[str]]
) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """
    Extracts features for all candidate pairs in batch.

    Args:
        s1_dict: Mapping of {entity_id: preprocessed_record}
        cand_dict: Mapping of {entity_id: preprocessed_record}
        candidate_pairs: Mapping of {s1_id: [candidate_ids]}

    Returns:
        (features_df, pair_keys)
        where pair_keys is a list of (s1_id, cand_id) tuples aligned with features_df rows.
    """
    feature_rows = []
    pair_keys = []

    for s1_id, cand_list in candidate_pairs.items():
        s1_rec = s1_dict.get(s1_id)
        if not s1_rec:
            continue

        for rank, cand_id in enumerate(cand_list, start=1):
            cand_rec = cand_dict.get(cand_id)
            if not cand_rec:
                continue

            feats = compute_pair_features(s1_rec, cand_rec, rank=rank)
            feature_rows.append(feats)
            pair_keys.append((s1_id, cand_id))

    if not feature_rows:
        features_df = pd.DataFrame(columns=FEATURE_NAMES)
    else:
        features_df = pd.DataFrame(feature_rows)[FEATURE_NAMES]

    return features_df, pair_keys
