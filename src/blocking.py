"""
Country-partitioned Address-First Blocking and Candidate Generation Engine.

Architecture (v2 — Address-First, No TF-IDF):
1. Dynamic Country Partitioning: Guarantees zero cross-country candidate pairing.
2. Multi-Channel Inverted Index: Indexes postal codes (weight=5), address tokens (weight=3),
   name tokens (weight=2), address numbers (weight=2) for high-recall retrieval.
3. Max-DF Filtering: Skips overly-common tokens to keep index focused and fast.
4. Top-50 Truncation by weighted hit count.
5. TSV Serializer: Emits competition-compliant `candidate_pairs.tsv`.

Key Improvement over v1:
- Removes TF-IDF entirely (saves ~4GB RAM on large partitions).
- Address tokens are now indexed, enabling matching for transliterated Indian names
  where the business name is in Hindi/Tamil script but the address is in English.
"""

import os
import csv
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import pandas as pd

from src.preprocess import (
    preprocess_record,
    preprocess_dataframe,
    extract_name_tokens,
)

# Common address words that appear in nearly every record and have low discriminative value
ADDRESS_STOPWORDS = {
    "street", "road", "avenue", "lane", "drive", "boulevard", "highway",
    "place", "court", "circle", "way", "suite", "floor", "building",
    "apartment", "unit", "room", "block", "sector", "phase", "plot",
    "near", "opp", "opposite", "behind", "next", "above", "below",
    "nagar", "colony", "enclave", "vihar", "puram", "town",
    "north", "south", "east", "west", "new", "old",
    "rue", "allee", "impasse", "chemin", "route", "pont",
    "main", "cross", "inner", "outer",
    "null", "none", "na",
}


class BlockingEngine:
    """
    Multi-channel inverted index candidate generator with address-first strategy.
    """

    def __init__(self, top_k: int = 50, max_df_ratio: float = 0.03):
        """
        Args:
            top_k: Maximum candidate entities per Source 1 entity (default: 50)
            max_df_ratio: Maximum document-frequency ratio; tokens appearing in more
                          than this fraction of candidates are excluded from the index
        """
        self.top_k = top_k
        self.max_df_ratio = max_df_ratio

    def build_inverted_index(
        self, cand_records: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, List[int]], int]:
        """
        Builds a multi-channel inverted index on candidate records.

        Channels:
        - post:{code}   → Postal/PIN code (high precision)
        - addr:{token}  → Significant address words, 4+ chars (city, street names)
        - name:{token}  → Core name tokens, 3+ chars
        - num:{number}  → Street/door numbers, 2+ digits

        Returns:
            (index, n_candidates): inverted index dict and count of candidates indexed
        """
        index: Dict[str, List[int]] = defaultdict(list)
        n_cands = len(cand_records)

        for idx, rec in enumerate(cand_records):
            # Index name tokens
            for token in rec.get("name_tokens", []):
                if len(token) >= 3:
                    index[f"name:{token}"].append(idx)

            # Index postal code
            postal = rec.get("postal_code", "")
            if postal and len(postal) >= 4:
                index[f"post:{postal}"].append(idx)

            # Index address numbers
            for num in rec.get("address_numbers", []):
                if len(num) >= 2:
                    index[f"num:{num}"].append(idx)

            # Index address tokens (city names, street names, etc.)
            for token in rec.get("address_tokens", []):
                if len(token) >= 4 and token.lower() not in ADDRESS_STOPWORDS:
                    index[f"addr:{token}"].append(idx)

        # Max-DF filtering: remove overly-common keys
        if n_cands > 100:
            max_postings = max(int(n_cands * self.max_df_ratio), 50)
            keys_to_remove = [k for k, v in index.items() if len(v) > max_postings]
            for k in keys_to_remove:
                del index[k]

        return dict(index), n_cands

    def query_inverted_index(
        self,
        s1_rec: Dict[str, Any],
        inverted_index: Dict[str, List[int]],
        max_from_index: int = 60
    ) -> List[int]:
        """
        Queries the multi-channel inverted index for candidates sharing signals with s1_rec.

        Weights:
        - Postal code match: +5 (strongest signal, very precise)
        - Address token match: +3 (city, street names — robust to transliteration)
        - Name token match: +2 (core name words)
        - Address number match: +2 (street/door numbers)

        Returns:
            List of candidate indices sorted by descending hit score
        """
        candidate_hits: Dict[int, int] = defaultdict(int)

        # Postal code (weight=5)
        postal = s1_rec.get("postal_code", "")
        if postal and len(postal) >= 4:
            key = f"post:{postal}"
            if key in inverted_index:
                for cand_idx in inverted_index[key]:
                    candidate_hits[cand_idx] += 5

        # Address tokens (weight=3)
        for token in s1_rec.get("address_tokens", []):
            if len(token) >= 4 and token.lower() not in ADDRESS_STOPWORDS:
                key = f"addr:{token}"
                if key in inverted_index:
                    for cand_idx in inverted_index[key]:
                        candidate_hits[cand_idx] += 3

        # Name tokens (weight=2)
        for token in s1_rec.get("name_tokens", []):
            if len(token) >= 3:
                key = f"name:{token}"
                if key in inverted_index:
                    for cand_idx in inverted_index[key]:
                        candidate_hits[cand_idx] += 2

        # Address numbers (weight=2)
        for num in s1_rec.get("address_numbers", []):
            if len(num) >= 2:
                key = f"num:{num}"
                if key in inverted_index:
                    for cand_idx in inverted_index[key]:
                        candidate_hits[cand_idx] += 2

        if not candidate_hits:
            return []

        # Sort by score descending, take top candidates
        sorted_hits = sorted(candidate_hits.items(), key=lambda x: x[1], reverse=True)
        return [cand_idx for cand_idx, _ in sorted_hits[:max_from_index]]

    def block_country_partition(
        self,
        s1_partition: List[Dict[str, Any]],
        cand_partition: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """
        Executes multi-channel candidate generation within a single country partition.

        Args:
            s1_partition: List of preprocessed S1 records for this country
            cand_partition: List of preprocessed S2 and S3 records for this country

        Returns:
            Mapping of {s1_entity_id: [candidate_entity_ids]}
        """
        partition_results: Dict[str, List[str]] = {}

        if not s1_partition:
            return partition_results

        # If there are no candidate records in this country partition, all S1 are singletons
        if not cand_partition:
            for s1 in s1_partition:
                partition_results[s1["entity_id"]] = []
            return partition_results

        # Build multi-channel inverted index
        inverted_index, n_cands = self.build_inverted_index(cand_partition)
        cand_ids = [c["entity_id"] for c in cand_partition]

        # Process S1 entities
        for s1_rec in s1_partition:
            s1_id = s1_rec["entity_id"]

            # Query inverted index (returns sorted candidate indices)
            cand_indices = self.query_inverted_index(
                s1_rec, inverted_index, max_from_index=self.top_k
            )

            if not cand_indices:
                partition_results[s1_id] = []
                continue

            top_cand_ids = [cand_ids[c_idx] for c_idx in cand_indices[:self.top_k]]
            partition_results[s1_id] = top_cand_ids

        return partition_results

    def generate_candidates(
        self,
        s1_df: pd.DataFrame,
        cand_df: pd.DataFrame
    ) -> Dict[str, List[str]]:
        """
        Partitions data by country and generates candidate pairs for all Source 1 records.

        Args:
            s1_df: DataFrame of Source 1 records (must contain entity_id, business_name, business_address, country)
            cand_df: DataFrame of Source 2 & 3 records

        Returns:
            Dictionary mapping source1_entity_id -> list of candidate entity_ids
        """
        # Preprocess if not already preprocessed
        if "clean_name" not in s1_df.columns:
            s1_processed = preprocess_dataframe(s1_df)
        else:
            s1_processed = s1_df

        if "clean_name" not in cand_df.columns:
            cand_processed = preprocess_dataframe(cand_df)
        else:
            cand_processed = cand_df

        # Partition by country
        s1_records = s1_processed.to_dict(orient="records")
        cand_records = cand_processed.to_dict(orient="records")

        s1_by_country: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        cand_by_country: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for rec in s1_records:
            country = str(rec.get("country", "")).strip().lower()
            s1_by_country[country].append(rec)

        for rec in cand_records:
            country = str(rec.get("country", "")).strip().lower()
            cand_by_country[country].append(rec)

        all_candidates: Dict[str, List[str]] = {}

        # Process each country partition
        for country, s1_partition in s1_by_country.items():
            cand_partition = cand_by_country.get(country, [])
            print(f"  Blocking [{country}]: {len(s1_partition)} S1 x {len(cand_partition)} candidates")
            partition_res = self.block_country_partition(s1_partition, cand_partition)
            all_candidates.update(partition_res)

        return all_candidates


def export_candidate_pairs(candidates: Dict[str, List[str]], output_path: str) -> None:
    """
    Serializes candidate dictionary to competition-compliant TSV file:
    source1_entity_id\tcandidate_entity_ids

    Args:
        candidates: Dict mapping source1_entity_id -> list of candidate IDs
        output_path: Path where candidate_pairs.tsv will be saved
    """
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for s1_id in sorted(candidates.keys()):
            cand_list = candidates[s1_id]
            # Ensure unique IDs and preserve ordering
            seen = set()
            unique_cands = [c for c in cand_list if not (c in seen or seen.add(c))]
            cand_str = ",".join(unique_cands)
            writer.writerow([s1_id, cand_str])


def measure_blocking_recall(
    candidate_pairs: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]]
) -> Dict[str, float]:
    """
    Measures blocking recall: what fraction of true matches appear in the candidate pool.

    Args:
        candidate_pairs: Blocking output {s1_id: [candidate_ids]}
        ground_truth: True matches {s1_id: {true_match_ids}}

    Returns:
        Dictionary with recall metrics
    """
    total_true = 0
    found_true = 0
    entities_with_matches = 0
    entities_perfect_recall = 0

    for s1_id, true_set in ground_truth.items():
        if not true_set:
            continue

        entities_with_matches += 1
        cand_set = set(candidate_pairs.get(s1_id, []))
        n_true = len(true_set)
        n_found = len(true_set & cand_set)

        total_true += n_true
        found_true += n_found

        if n_found == n_true:
            entities_perfect_recall += 1

    recall = found_true / total_true if total_true > 0 else 0.0
    entity_recall = entities_perfect_recall / entities_with_matches if entities_with_matches > 0 else 0.0

    return {
        "pair_recall": recall,
        "entity_perfect_recall": entity_recall,
        "total_true_pairs": total_true,
        "found_true_pairs": found_true,
        "entities_with_matches": entities_with_matches,
        "entities_perfect_recall": entities_perfect_recall,
    }


def run_blocking_pipeline(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    output_path: str,
    top_k: int = 50
) -> Dict[str, List[str]]:
    """
    High-level entry point to run candidate blocking from TSV file paths.

    Args:
        s1_path: Path to Source 1 TSV
        s2_path: Path to Source 2 TSV
        s3_path: Path to Source 3 TSV
        output_path: Destination path for candidate_pairs.tsv
        top_k: Maximum candidate pool size per S1 entity

    Returns:
        Generated candidate dictionary
    """
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    df_s2 = pd.read_csv(s2_path, sep="\t", dtype=str, keep_default_na=False)
    df_s3 = pd.read_csv(s3_path, sep="\t", dtype=str, keep_default_na=False)

    cand_df = pd.concat([df_s2, df_s3], ignore_index=True)

    blocker = BlockingEngine(top_k=top_k)
    candidates = blocker.generate_candidates(df_s1, cand_df)

    export_candidate_pairs(candidates, output_path)
    return candidates
