"""
Test script to demonstrate aspect normalization functionality.

This script shows example inputs and outputs of the aspect normalization system.
"""

from aspect_normalization import (
    normalize_aspects,
    parse_aspects_string,
    normalize_aspects_column,
    get_aspect_statistics,
    validate_normalization,
)
from aspect_mappings import get_canonical_aspect, CANONICAL_ASPECTS
import pandas as pd


def test_parse_aspects():
    """Test parsing of Azure aspect strings."""
    print("=" * 60)
    print("Testing Aspect String Parsing")
    print("=" * 60)
    
    test_cases = [
        "baño (negative): sucio, mal olor",
        "aire acondicionado (positive): funciona bien | habitación (negative): pequeña",
        "servicio (neutral): regular",
        None,
        "",
    ]
    
    for test in test_cases:
        parsed = parse_aspects_string(test)
        print(f"Input: {test}")
        print(f"Parsed: {parsed}")
        print()


def test_canonical_mapping():
    """Test canonical aspect mapping."""
    print("=" * 60)
    print("Testing Canonical Aspect Mapping")
    print("=" * 60)
    
    test_cases = [
        ("baño", "bathroom_cleanliness"),
        ("baños", "bathroom_cleanliness"),
        ("aire", "air_conditioning"),
        ("aire acondicionado", "air_conditioning"),
        ("cama", "bed_comfort"),
        ("camas", "bed_comfort"),
        ("aseo", "bathroom_cleanliness"),
        ("atención aseo", "bathroom_cleanliness"),
        ("unknown_aspect_xyz", "unclassified"),
        ("", "unclassified"),
    ]
    
    for raw, expected in test_cases:
        result = get_canonical_aspect(raw)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{raw}' → '{result}' (expected: '{expected}')")


def test_normalize_aspects():
    """Test full aspect normalization."""
    print("=" * 60)
    print("Testing Full Aspect Normalization")
    print("=" * 60)
    
    test_cases = [
        "baño (negative): sucio",
        "aire acondicionado (positive): funciona | habitación (negative): pequeña",
        "servicio (neutral): regular | precio (negative): caro",
        None,
    ]
    
    for test in test_cases:
        result = normalize_aspects(test, keep_original=True)
        print(f"Input: {test}")
        print(f"Original aspects: {result['original']}")
        print(f"Normalized: {result['normalized']}")
        print(f"Normalized string: {result['normalized_str']}")
        print()


def test_dataframe_normalization():
    """Test DataFrame normalization."""
    print("=" * 60)
    print("Testing DataFrame Normalization")
    print("=" * 60)
    
    # Create sample DataFrame
    df = pd.DataFrame({
        "reviewId": ["r1", "r2", "r3", "r4"],
        "aspects": [
            "baño (negative): sucio",
            "aire acondicionado (positive): funciona | habitación (negative): pequeña",
            "servicio (neutral): regular",
            None,
        ],
    })
    
    print("Original DataFrame:")
    print(df)
    print()
    
    # Normalize
    df_normalized = normalize_aspects_column(df, aspect_col="aspects", output_col="aspect_normalized")
    
    print("Normalized DataFrame:")
    print(df_normalized[["reviewId", "aspects", "aspect_normalized"]])
    print()
    
    # Get statistics
    stats = get_aspect_statistics(df_normalized, aspect_col="aspect_normalized")
    print("Aspect Statistics:")
    print(stats)
    print()
    
    # Validate
    validation = validate_normalization(df_normalized, aspect_col="aspect_normalized")
    print("Validation Result:")
    print(f"  Valid: {validation['valid']}")
    print(f"  Invalid aspects: {validation.get('invalid_aspects', set())}")
    print(f"  Unclassified count: {validation['unclassified_count']}")
    print(f"  Total unique aspects: {validation['total_unique_aspects']}")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ASPECT NORMALIZATION SYSTEM - TEST SUITE")
    print("=" * 60 + "\n")
    
    test_parse_aspects()
    test_canonical_mapping()
    test_normalize_aspects()
    test_dataframe_normalization()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
    print(f"\nCanonical aspects available: {len(CANONICAL_ASPECTS)}")
    print(f"Categories: {', '.join(sorted(CANONICAL_ASPECTS))}")


if __name__ == "__main__":
    main()

