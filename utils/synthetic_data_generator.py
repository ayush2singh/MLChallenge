"""
Synthetic Data Generator for Business Entity Resolution Challenge.
Generates realistic multi-source TSV files mimicking the problem statement:
- Source 1: Deduplicated reference source
- Source 2 & 3: Noisy data sources with typos, abbreviations, landmark addresses, missing fields
- Ground Truth: Source 1 entity -> matching S2/S3 entity IDs (including singletons)
- Countries: US, India (in train), and France (in test)
"""

import os
import random
import csv
from typing import List, Tuple, Dict, Set


LEGAL_SUFFIX_MAP = {
    "Corporation": ["Corp.", "Corp", "Corporation", "Inc."],
    "Inc.": ["Inc", "Incorporated", "Inc."],
    "LLC": ["L.L.C.", "LLC", "Limited Liability Company"],
    "Private Limited": ["Pvt. Ltd.", "Pvt Ltd", "Private Limited", "Ltd."],
    "Limited": ["Ltd.", "Ltd", "Limited"],
    "SA": ["S.A.", "SA", "Société Anonyme"],
    "SAS": ["S.A.S.", "SAS", "Société par actions simplifiée"]
}

STREET_SUFFIX_MAP = {
    "Street": ["St.", "St", "Street"],
    "Road": ["Rd.", "Rd", "Road"],
    "Avenue": ["Ave.", "Ave", "Avenue"],
    "Boulevard": ["Blvd.", "Blvd", "Boulevard"],
    "Rue": ["Rue", "R."],
}

SEED_ENTITIES = [
    # US
    {"name": "Alpha Apex Technologies Inc.", "address": "100 Innovation Way, Suite 300, San Jose, CA 95134", "country": "US"},
    {"name": "Metro Beacon Healthcare LLC", "address": "450 Lexington Ave, New York, NY 10017", "country": "US"},
    {"name": "Pinnacle Financial Advisors Corp.", "address": "77 W Wacker Dr, Chicago, IL 60601", "country": "US"},
    {"name": "Summit Peak Logistics Corporation", "address": "1200 Harbor Blvd, Long Beach, CA 90802", "country": "US"},
    {"name": "Evergreen Organic Market LLC", "address": "890 Market St, San Francisco, CA 94102", "country": "US"},
    {"name": "Horizon Clean Energy Inc.", "address": "333 S Grand Ave, Los Angeles, CA 90071", "country": "US"},
    {"name": "Vanguard Digital Media Corp.", "address": "500 Boylston St, Boston, MA 02116", "country": "US"},
    {"name": "Cascade Mountain Brewing Co.", "address": "1420 5th Ave, Seattle, WA 98101", "country": "US"},
    {"name": "Redwood Real Estate Partners LLC", "address": "2000 Avenue of the Stars, Los Angeles, CA 90067", "country": "US"},
    {"name": "Sterling Automotive Solutions Inc.", "address": "100 Renaissance Center, Detroit, MI 48243", "country": "US"},

    # India
    {"name": "Bharat Dynamic Systems Pvt. Ltd.", "address": "Plot 42, Electronics City Phase 1, Bangalore, Karnataka 560100", "country": "India"},
    {"name": "Kalyani Infotech Solutions Limited", "address": "Tower B, Cyber Gateway, Hitec City, Hyderabad, Telangana 500081", "country": "India"},
    {"name": "Venkateshwara Textiles & Garments", "address": "Shop 14, Commercial Complex, Near SBI ATM, MG Road, Pune, Maharashtra 411001", "country": "India"},
    {"name": "Shree Ganesh Trading Corporation", "address": "3rd Floor, APMC Market Yard, Vashi, Navi Mumbai, Maharashtra 400703", "country": "India"},
    {"name": "Himalayan Herbal Wellness Pvt. Ltd.", "address": "Sector 62, Opp Metro Station, Noida, Uttar Pradesh 201309", "country": "India"},
    {"name": "Surya Power & Infrastructure Ltd.", "address": "15 Barakhamba Road, Connaught Place, New Delhi, Delhi 110001", "country": "India"},
    {"name": "Godavari Logistics & Supply Chain", "address": "Gala No 5, Godown Area, Near Toll Plaza, Bhiwandi, Maharashtra 421302", "country": "India"},
    {"name": "Chola Agro Industries Limited", "address": "Door No 22, Old Trunk Road, Guindy, Chennai, Tamil Nadu 600032", "country": "India"},

    # France (in test set)
    {"name": "Atelier Lumière Artisanat SAS", "address": "24 Rue de Rivoli, 75004 Paris", "country": "France"},
    {"name": "Bordeaux Vignobles & Distribution SA", "address": "12 Cours de l'Intendance, 33000 Bordeaux", "country": "France"},
    {"name": "Lyon Mécanique Industrielle SAS", "address": "45 Rue Garibaldi, 69006 Lyon", "country": "France"},
    {"name": "Provence Lavande & Parfums", "address": "8 Place aux Herbes, 13100 Aix-en-Provence", "country": "France"},
]


def perturb_name(name: str) -> str:
    """Applies common entity resolution name variations."""
    res = name
    
    # 1. Legal suffix variation
    for canonical, variants in LEGAL_SUFFIX_MAP.items():
        if canonical in res:
            res = res.replace(canonical, random.choice(variants))
            break
            
    # 2. Punctuation (& vs and)
    if " & " in res and random.random() < 0.5:
        res = res.replace(" & ", " and ")
    elif " and " in res and random.random() < 0.5:
        res = res.replace(" and ", " & ")
        
    # 3. Typo simulation (character swap or drop)
    if random.random() < 0.25 and len(res) > 8:
        pos = random.randint(3, len(res) - 4)
        # Drop a character or swap adjacent
        if random.random() < 0.5:
            res = res[:pos] + res[pos+1:]
        else:
            res = res[:pos] + res[pos+1] + res[pos] + res[pos+2:]
            
    return res


def perturb_address(addr: str, country: str) -> str:
    """Applies realistic address noise according to country conventions."""
    res = addr
    
    # Street suffix variation
    for canonical, variants in STREET_SUFFIX_MAP.items():
        if f" {canonical}" in res:
            res = res.replace(f" {canonical}", f" {random.choice(variants)}")
            break
            
    # India specific noise: landmark insertion or postal code drop
    if country == "India":
        if random.random() < 0.3 and "Near" not in res and "Opp" not in res:
            landmarks = ["Near HDFC Bank", "Opposite Bus Stand", "Behind Petrol Pump", "Near City Post Office"]
            tokens = res.split(",")
            if len(tokens) > 2:
                tokens.insert(2, random.choice(landmarks))
                res = ",".join(tokens)
        if random.random() < 0.2:
            # Drop pincode (last 6 digits)
            import re
            res = re.sub(r"\b\d{6}\b", "", res).strip(", ")
            
    # US specific noise: suite variation or abbreviation
    elif country == "US":
        if "Suite" in res and random.random() < 0.5:
            res = res.replace("Suite", "Ste.")
            
    # Component drop
    parts = [p.strip() for p in res.split(",") if p.strip()]
    if len(parts) > 3 and random.random() < 0.2:
        # Drop one middle component
        drop_idx = random.randint(1, len(parts) - 2)
        parts.pop(drop_idx)
        res = ", ".join(parts)
        
    return res


def generate_benchmark_dataset(
    output_dir: str,
    num_train_s1: int = 200,
    num_test_s1: int = 100,
    seed: int = 42
):
    """Generates synthetic train and test datasets."""
    random.seed(seed)
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 1. Generate Train Set (US and India only)
    train_seeds = [e for e in SEED_ENTITIES if e["country"] in ["US", "India"]]
    
    s1_train_records = []
    s2_train_records = []
    s3_train_records = []
    ground_truth = []
    
    s2_counter = 1
    s3_counter = 1
    
    for i in range(1, num_train_s1 + 1):
        s1_id = f"S1-{i:05d}"
        template = random.choice(train_seeds)
        country = template["country"]
        base_name = f"{template['name']} Branch {i}" if i > len(train_seeds) else template["name"]
        base_addr = template["address"]
        
        s1_train_records.append({
            "entity_id": s1_id,
            "business_name": base_name,
            "business_address": base_addr,
            "country": country
        })
        
        # Decide matches:
        # 35% singletons (0 matches)
        # 40% 1 match (either S2 or S3)
        # 25% multi matches (both S2 and S3, or multiple S2/S3)
        roll = random.random()
        matched_ids = []
        
        if roll >= 0.35: # Has matches
            if roll < 0.75: # 1 match
                if random.random() < 0.5:
                    target_id = f"S2-{s2_counter:05d}"
                    s2_counter += 1
                    s2_train_records.append({
                        "entity_id": target_id,
                        "business_name": perturb_name(base_name),
                        "business_address": perturb_address(base_addr, country),
                        "country": country
                    })
                    matched_ids.append(target_id)
                else:
                    target_id = f"S3-{s3_counter:05d}"
                    s3_counter += 1
                    s3_train_records.append({
                        "entity_id": target_id,
                        "business_name": perturb_name(base_name),
                        "business_address": perturb_address(base_addr, country),
                        "country": country
                    })
                    matched_ids.append(target_id)
            else: # Multi matches
                num_matches = random.randint(2, 3)
                for _ in range(num_matches):
                    if random.random() < 0.5:
                        target_id = f"S2-{s2_counter:05d}"
                        s2_counter += 1
                        s2_train_records.append({
                            "entity_id": target_id,
                            "business_name": perturb_name(base_name),
                            "business_address": perturb_address(base_addr, country),
                            "country": country
                        })
                        matched_ids.append(target_id)
                    else:
                        target_id = f"S3-{s3_counter:05d}"
                        s3_counter += 1
                        s3_train_records.append({
                            "entity_id": target_id,
                            "business_name": perturb_name(base_name),
                            "business_address": perturb_address(base_addr, country),
                            "country": country
                        })
                        matched_ids.append(target_id)
                        
        ground_truth.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(matched_ids)
        })

    # Add some random distractors to S2 and S3 (unmatched entities)
    for _ in range(50):
        target_id = f"S2-{s2_counter:05d}"
        s2_counter += 1
        template = random.choice(train_seeds)
        s2_train_records.append({
            "entity_id": target_id,
            "business_name": f"Distractor {perturb_name(template['name'])}",
            "business_address": perturb_address(template["address"], template["country"]),
            "country": template["country"]
        })
        
    for _ in range(50):
        target_id = f"S3-{s3_counter:05d}"
        s3_counter += 1
        template = random.choice(train_seeds)
        s3_train_records.append({
            "entity_id": target_id,
            "business_name": f"Standalone {perturb_name(template['name'])}",
            "business_address": perturb_address(template["address"], template["country"]),
            "country": template["country"]
        })

    # 2. Generate Test Set (US, India, AND France)
    s1_test_records = []
    s2_test_records = []
    s3_test_records = []
    
    s2_test_counter = 1
    s3_test_counter = 1
    
    for i in range(1, num_test_s1 + 1):
        s1_id = f"S1-{i+10000:05d}"
        template = random.choice(SEED_ENTITIES) # Includes France!
        country = template["country"]
        base_name = f"{template['name']} Unit {i}"
        base_addr = template["address"]
        
        s1_test_records.append({
            "entity_id": s1_id,
            "business_name": base_name,
            "business_address": base_addr,
            "country": country
        })
        
        # Test also contains matches in S2/S3
        if random.random() > 0.3:
            if random.random() < 0.6:
                target_id = f"S2-{s2_test_counter+10000:05d}"
                s2_test_counter += 1
                s2_test_records.append({
                    "entity_id": target_id,
                    "business_name": perturb_name(base_name),
                    "business_address": perturb_address(base_addr, country),
                    "country": country
                })
            else:
                target_id = f"S3-{s3_test_counter+10000:05d}"
                s3_test_counter += 1
                s3_test_records.append({
                    "entity_id": target_id,
                    "business_name": perturb_name(base_name),
                    "business_address": perturb_address(base_addr, country),
                    "country": country
                })

    # Shuffle records to avoid positional bias
    random.shuffle(s2_train_records)
    random.shuffle(s3_train_records)
    random.shuffle(s2_test_records)
    random.shuffle(s3_test_records)

    def write_tsv(filepath: str, fieldnames: List[str], rows: List[dict]):
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    fieldnames = ["entity_id", "business_name", "business_address", "country"]
    write_tsv(os.path.join(train_dir, "train_source1.tsv"), fieldnames, s1_train_records)
    write_tsv(os.path.join(train_dir, "train_source2.tsv"), fieldnames, s2_train_records)
    write_tsv(os.path.join(train_dir, "train_source3.tsv"), fieldnames, s3_train_records)
    write_tsv(os.path.join(train_dir, "train_ground_truth.tsv"), ["source1_entity_id", "matched_entity_ids"], ground_truth)

    write_tsv(os.path.join(test_dir, "test_source1.tsv"), fieldnames, s1_test_records)
    write_tsv(os.path.join(test_dir, "test_source2.tsv"), fieldnames, s2_test_records)
    write_tsv(os.path.join(test_dir, "test_source3.tsv"), fieldnames, s3_test_records)
    
    print(f"Generated synthetic dataset in '{output_dir}':")
    print(f"  Train: S1={len(s1_train_records)}, S2={len(s2_train_records)}, S3={len(s3_train_records)}, GT={len(ground_truth)}")
    print(f"  Test:  S1={len(s1_test_records)}, S2={len(s2_test_records)}, S3={len(s3_test_records)}")


if __name__ == "__main__":
    generate_benchmark_dataset("dataset", num_train_s1=200, num_test_s1=100)
