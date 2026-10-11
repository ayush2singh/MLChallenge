"""
Preprocessing and text normalization pipeline for Business Entity Resolution.

Key Capabilities:
1. NFKD Unicode Normalization: Strips diacritics and accents (vital for French test entities).
2. Multilingual Legal Suffix Stripper: Removes company forms across US, India, and France.
3. Address Token & Postal/PIN Code Extractor: Handles US ZIP, India 6-digit PIN, France 5-digit code.
4. Fast Out-of-Core Batch & Streaming Processing for large tabular TSVs.
"""

import re
import unicodedata
from typing import Dict, List, Optional, Set, Any, Generator
import pandas as pd


# Stopwords that carry negligible discriminative power in business names
BUSINESS_STOPWORDS: Set[str] = {
    "and", "the", "of", "in", "for", "at", "by", "from", "with", "a", "an", "on", "to"
}

# Standard street suffix mapping
STREET_SUFFIX_MAP: Dict[str, str] = {
    "st": "street",
    "str": "street",
    "rd": "road",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "hwy": "highway",
    "pkg": "parkway",
    "pkwy": "parkway",
    "ste": "suite",
    "fl": "floor",
    "bldg": "building",
    "apt": "apartment",
    "r": "rue",
    "bd": "boulevard",
    "pl": "place",
    "all": "allee",
}

# Legal suffixes compiled patterns ordered from longest to shortest
LEGAL_SUFFIX_PATTERNS = [
    # Multi-word patterns (India / US / France)
    r"\bprivate\s+limited\b",
    r"\bpvt\s+limited\b",
    r"\bpvt\s+ltd\b",
    r"\bpvt\b",
    r"\bcorporation\b",
    r"\bincorporated\b",
    r"\blimited\s+liability\s+company\b",
    r"\bpublic\s+limited\b",
    r"\bsociete\s+par\s+actions\s+simplifiee\b",
    r"\bsociete\s+anonyme\b",
    r"\bsociete\s+a\s+responsabilite\s+limitee\b",
    r"\bentreprise\s+unipersonnelle\s+a\s+responsabilite\s+limitee\b",
    r"\bsociete\s+en\s+nom\s+collectif\b",
    # Single-word / acronym patterns
    r"\bcorp\b",
    r"\binc\b",
    r"\bllc\b",
    r"\bllp\b",
    r"\bltd\b",
    r"\blimited\b",
    r"\bco\b",
    r"\bcompany\b",
    r"\bsas\b",
    r"\bsarl\b",
    r"\beurl\b",
    r"\bsnc\b",
    r"\bsa\b",
]

# Combined regex matching legal suffix at the end of string
RE_LEGAL_SUFFIX_END = re.compile(
    r"(?:(?:\s+|^)(?:" + "|".join(LEGAL_SUFFIX_PATTERNS) + r"))+\s*$",
    re.IGNORECASE
)

# Postal code patterns
RE_INDIA_PIN = re.compile(r"\b([1-9]\d{5})\b")
RE_US_ZIP = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
RE_FRANCE_POSTAL = re.compile(r"\b(\d{5})\b")
RE_GENERAL_POSTAL = re.compile(r"\b(\d{5,6})\b")
RE_NUMBERS = re.compile(r"\b\d+\b")
RE_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9\s]")
RE_EXTRA_SPACES = re.compile(r"\s+")


def normalize_text(text: Optional[str]) -> str:
    """
    Normalizes arbitrary text string:
    - Handles null/NaN safely
    - Strips accents and diacritics via NFKD Unicode normalization (e.g., Lumière -> Lumiere)
    - Expands ampersand '&' to ' and '
    - Converts to lowercase
    - Replaces non-alphanumeric characters with spaces
    - Collapses consecutive whitespaces
    
    Args:
        text: Input string or None
        
    Returns:
        Clean normalized string
    """
    if text is None or not isinstance(text, str):
        return ""
        
    # NFKD decomposition to strip accents
    normalized = unicodedata.normalize("NFKD", text)
    ascii_bytes = normalized.encode("ASCII", "ignore")
    cleaned = ascii_bytes.decode("utf-8")
    
    # Expand ampersand
    cleaned = cleaned.replace("&", " and ")
    
    # Lowercase and clean non-alphanumerics
    cleaned = cleaned.lower()
    cleaned = RE_NON_ALPHANUMERIC.sub(" ", cleaned)
    cleaned = RE_EXTRA_SPACES.sub(" ", cleaned).strip()
    return cleaned


def strip_legal_suffixes(name: str) -> str:
    """
    Removes legal entity designations (e.g. Inc, LLC, Corp, Pvt Ltd, SA, SAS, SARL)
    from the trailing end of a business name.
    
    Args:
        name: Normalized business name
        
    Returns:
        Core business name without legal suffixes
    """
    if not name:
        return ""
        
    stripped = name
    # Strip iteratively in case multiple legal tokens exist (e.g. "Acme Co Inc")
    for _ in range(3):
        new_stripped = RE_LEGAL_SUFFIX_END.sub("", stripped).strip()
        if new_stripped == stripped:
            break
        stripped = new_stripped
        
    # If stripping resulted in empty string, fallback to original name
    return stripped if stripped else name


def extract_postal_code(address: str, country: Optional[str] = None) -> str:
    """
    Extracts postal or PIN code from address based on country format.
    
    Args:
        address: Raw or normalized address string
        country: Country label ('US', 'India', 'France', or None)
        
    Returns:
        Extracted postal code string or empty string
    """
    if not address:
        return ""
        
    country_lower = country.lower().strip() if country else ""
    
    if "india" in country_lower:
        match = RE_INDIA_PIN.search(address)
        if match:
            return match.group(1)
            
    elif "us" in country_lower or "united states" in country_lower:
        match = RE_US_ZIP.search(address)
        if match:
            return match.group(1)
            
    elif "france" in country_lower:
        match = RE_FRANCE_POSTAL.search(address)
        if match:
            return match.group(1)
            
    # Generic fallback: look for 5 or 6 digit number
    match = RE_GENERAL_POSTAL.search(address)
    if match:
        return match.group(1)
        
    return ""


def extract_address_tokens(address: Optional[str], country: Optional[str] = None) -> Dict[str, Any]:
    """
    Extracts structured address components:
    - Normalized address text
    - Standardized street words
    - Postal code
    - Numerical identifiers (door numbers, plot numbers)
    - Significant token set
    
    Args:
        address: Raw address string
        country: Country name
        
    Returns:
        Dictionary of extracted address components
    """
    if not address or not isinstance(address, str):
        return {
            "clean_address": "",
            "postal_code": "",
            "numbers": [],
            "tokens": []
        }
        
    postal = extract_postal_code(address, country)
    norm = normalize_text(address)
    
    words = norm.split()
    standardized_words = [STREET_SUFFIX_MAP.get(w, w) for w in words]
    clean_address = " ".join(standardized_words)
    
    numbers = RE_NUMBERS.findall(clean_address)
    tokens = [w for w in standardized_words if len(w) >= 2 and w not in BUSINESS_STOPWORDS]
    
    return {
        "clean_address": clean_address,
        "postal_code": postal,
        "numbers": numbers,
        "tokens": tokens
    }


def extract_name_tokens(name: Optional[str]) -> List[str]:
    """
    Extracts significant name tokens after legal suffix removal and normalization.
    Includes prefix tokens (3 and 4 char prefixes) for high typo resilience.
    
    Args:
        name: Business name string
        
    Returns:
        List of significant tokens (length >= 2, non-stopwords) and prefix signatures
    """
    if not name or not isinstance(name, str):
        return []
        
    norm = normalize_text(name)
    core = strip_legal_suffixes(norm)
    words = [t for t in core.split() if len(t) >= 2 and t not in BUSINESS_STOPWORDS]
    
    tokens = list(words)
    # Add prefix tokens for typo resilience
    for w in words:
        if len(w) >= 4:
            tokens.append(f"pfx:{w[:3]}")
            tokens.append(f"pfx:{w[:4]}")
    return tokens


def preprocess_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocesses a single raw entity record dictionary.
    
    Expected keys: 'entity_id', 'business_name', 'business_address', 'country'
    """
    name = record.get("business_name", "")
    addr = record.get("business_address", "")
    country = str(record.get("country", "")).strip()
    
    norm_name = normalize_text(name)
    core_name = strip_legal_suffixes(norm_name)
    addr_info = extract_address_tokens(addr, country)
    name_tokens = extract_name_tokens(name)
    
    return {
        "entity_id": record.get("entity_id", ""),
        "country": country,
        "raw_name": name or "",
        "clean_name": core_name,
        "full_clean_name": norm_name,
        "clean_address": addr_info["clean_address"],
        "postal_code": addr_info["postal_code"],
        "address_numbers": addr_info["numbers"],
        "address_tokens": addr_info["tokens"],
        "name_tokens": name_tokens,
        # Search text combining core name, postal code, and significant address tokens
        "search_text": f"{core_name} {addr_info['postal_code']} {' '.join(addr_info['numbers'][:3])}".strip()
    }


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Batch processes a pandas DataFrame of business entities.
    
    Args:
        df: DataFrame containing ['entity_id', 'business_name', 'business_address', 'country']
        
    Returns:
        DataFrame with added normalized columns
    """
    records = df.to_dict(orient="records")
    processed = [preprocess_record(r) for r in records]
    return pd.DataFrame(processed)


def stream_tsv_chunks(file_path: str, chunk_size: int = 100_000) -> Generator[pd.DataFrame, None, None]:
    """
    Streams a large TSV file in out-of-core chunks with automated preprocessing.
    
    Args:
        file_path: Absolute or relative path to TSV file
        chunk_size: Number of records per chunk
        
    Yields:
        Preprocessed pd.DataFrame per chunk
    """
    for chunk in pd.read_csv(file_path, sep="\t", chunksize=chunk_size, dtype=str, keep_default_na=False):
        yield preprocess_dataframe(chunk)
