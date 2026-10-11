# Project: Large-Scale Business Entity Resolution (72h Challenge)

## Context
Multi-source Record Linkage across 3 data sources (Source 1 = deduplicated reference; Sources 2 & 3 = noisy inputs).
Target scale is 50-100 GB tab-separated data, executed on Kaggle (30 GB RAM, CPU/GPU) with local development harness.
Deadline: 72 hours.

## Objectives
1. Build out-of-core blocking and candidate generation pipeline that scales to 100 GB without memory overflow.
2. Build pairwise matching model (LightGBM) optimized specifically for Macro $F_{0.5}$ metric (precision-heavy, singleton-dominant).
3. Generate valid `matching_results.tsv` and `candidate_pairs.tsv` satisfying all rules.
4. Provide a 1-click standalone Kaggle notebook and the final submission package zip.
