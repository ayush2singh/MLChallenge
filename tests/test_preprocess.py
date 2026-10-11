"""
Unit tests for src/preprocess.py
"""

import pytest
import pandas as pd
import tempfile
import os
from src.preprocess import (
    normalize_text,
    strip_legal_suffixes,
    extract_postal_code,
    extract_address_tokens,
    extract_name_tokens,
    preprocess_record,
    preprocess_dataframe,
    stream_tsv_chunks,
)


def test_normalize_text_french_diacritics():
    """Verify NFKD decomposition strips accents from French words."""
    assert normalize_text("Atelier Lumière Artisanat") == "atelier lumiere artisanat"
    assert normalize_text("Lyon Mécanique Industrielle") == "lyon mecanique industrielle"
    assert normalize_text("Café & Thé Français") == "cafe and the francais"
    assert normalize_text("Château d'Yquem") == "chateau d yquem"


def test_normalize_text_ampersand_and_punctuation():
    """Verify ampersands are expanded and special characters stripped."""
    assert normalize_text("Venkateshwara Textiles & Garments") == "venkateshwara textiles and garments"
    assert normalize_text("Smith, Jones & Co. (USA)") == "smith jones and co usa"


def test_normalize_text_empty_and_null():
    """Verify null and empty inputs return empty strings without raising errors."""
    assert normalize_text("") == ""
    assert normalize_text(None) == ""
    assert normalize_text("   ") == ""


def test_strip_legal_suffixes_us():
    """Verify stripping US legal suffixes at end of name."""
    assert strip_legal_suffixes("alpha apex technologies inc") == "alpha apex technologies"
    assert strip_legal_suffixes("metro beacon healthcare llc") == "metro beacon healthcare"
    assert strip_legal_suffixes("pinnacle financial advisors corp") == "pinnacle financial advisors"
    assert strip_legal_suffixes("summit peak logistics corporation") == "summit peak logistics"
    assert strip_legal_suffixes("redwood partners limited liability company") == "redwood partners"


def test_strip_legal_suffixes_india():
    """Verify stripping Indian legal suffixes at end of name."""
    assert strip_legal_suffixes("bharat dynamic systems pvt ltd") == "bharat dynamic systems"
    assert strip_legal_suffixes("kalyani infotech solutions limited") == "kalyani infotech solutions"
    assert strip_legal_suffixes("godavari logistics private limited") == "godavari logistics"
    assert strip_legal_suffixes("surya power and infrastructure ltd") == "surya power and infrastructure"


def test_strip_legal_suffixes_france():
    """Verify stripping French legal designations at end of name."""
    assert strip_legal_suffixes("atelier lumiere artisanat sas") == "atelier lumiere artisanat"
    assert strip_legal_suffixes("bordeaux distribution sa") == "bordeaux distribution"
    assert strip_legal_suffixes("lyon mecanique industrielle sarl") == "lyon mecanique industrielle"
    assert strip_legal_suffixes("paris commerce eurl") == "paris commerce"


def test_strip_legal_suffixes_embedded_word():
    """Verify suffixes in middle or beginning of name are preserved."""
    # 'incorporated' in middle should not be stripped
    res = strip_legal_suffixes("incorporated logistics co")
    assert "incorporated" in res
    assert not res.endswith("co")
    
    # Standalone suffix should not be completely stripped to blank
    assert strip_legal_suffixes("inc") == "inc"
    assert strip_legal_suffixes("llc") == "llc"


def test_extract_postal_code():
    """Verify postal / PIN code extraction across countries."""
    # US 5-digit ZIP
    assert extract_postal_code("100 Innovation Way, San Jose, CA 95134", "US") == "95134"
    assert extract_postal_code("450 Lexington Ave, New York, NY 10017-1234", "US") == "10017"
    
    # India 6-digit PIN
    assert extract_postal_code("Plot 42, Electronics City, Bangalore 560100", "India") == "560100"
    assert extract_postal_code("MG Road, Pune 411001", "India") == "411001"
    
    # France 5-digit code
    assert extract_postal_code("24 Rue de Rivoli, 75004 Paris", "France") == "75004"
    assert extract_postal_code("12 Cours de l'Intendance, 33000 Bordeaux", "France") == "33000"


def test_extract_address_tokens():
    """Verify structured address token extraction."""
    res = extract_address_tokens("100 Innovation Way, Ste 300, San Jose, CA 95134", "US")
    assert res["postal_code"] == "95134"
    assert "100" in res["numbers"]
    assert "300" in res["numbers"]
    assert "street" not in res["clean_address"] or "suite" in res["clean_address"]
    
    # Empty address
    empty_res = extract_address_tokens("", "US")
    assert empty_res["postal_code"] == ""
    assert empty_res["numbers"] == []


def test_preprocess_record():
    """Verify end-to-end preprocessing of a single record dictionary."""
    rec = {
        "entity_id": "S1-00001",
        "business_name": "Atelier Lumière Artisanat SAS",
        "business_address": "24 Rue de Rivoli, 75004 Paris",
        "country": "France"
    }
    processed = preprocess_record(rec)
    assert processed["entity_id"] == "S1-00001"
    assert processed["clean_name"] == "atelier lumiere artisanat"
    assert processed["postal_code"] == "75004"
    assert "24" in processed["address_numbers"]
    assert "atelier" in processed["name_tokens"]


def test_stream_tsv_chunks():
    """Verify streaming chunks works with temporary TSV file."""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".tsv", newline="", encoding="utf-8") as f:
        f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        f.write("S1-001\tCorp A Inc.\t123 Main St\tUS\n")
        f.write("S1-002\tPvt B Ltd\tSector 62, Noida 201309\tIndia\n")
        temp_path = f.name
        
    try:
        chunks = list(stream_tsv_chunks(temp_path, chunk_size=1))
        assert len(chunks) == 2
        assert chunks[0].iloc[0]["clean_name"] == "corp a"
        assert chunks[1].iloc[0]["postal_code"] == "201309"
    finally:
        os.remove(temp_path)
