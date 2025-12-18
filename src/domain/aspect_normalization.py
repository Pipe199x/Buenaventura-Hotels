"""
Aspect normalization module for hotel reviews - Thematic categorization.

This module provides functions to normalize aspect values extracted from Azure Cognitive Services
into thematic categories for consistent analytical reporting.

Two-level model:
- aspect_raw: Original extracted text from Azure
- aspect_theme: Normalized thematic category for analytics

NO generic categories (other, unclassified, misc) are allowed.
All aspects must map to meaningful thematic categories.
"""

import re
from typing import Optional, List, Set, Dict
import pandas as pd

from .aspect_mappings import get_thematic_aspect, THEMATIC_CATEGORIES


def parse_aspects_string(aspects_str: Optional[str]) -> List[str]:
    """
    Parse the aspects string from Azure format into individual aspect names.
    
    Azure format: "target (sentiment): assessment | target2 (sentiment2): assessment2"
    
    Args:
        aspects_str: Raw aspects string from Azure Cognitive Services
        
    Returns:
        List of extracted aspect names (targets) - these are aspect_raw values
    """
    if not aspects_str or pd.isna(aspects_str):
        return []
    
    aspects_str = str(aspects_str)
    
    # Split by pipe separator
    parts = [p.strip() for p in aspects_str.split("|")]
    
    extracted_aspects = []
    for part in parts:
        # Extract target from format: "target (sentiment): assessment"
        # Pattern: everything before the first "("
        match = re.match(r"^([^(]+)", part.strip())
        if match:
            target = match.group(1).strip()
            if target:
                extracted_aspects.append(target)
    
    return extracted_aspects


def normalize_aspects_to_themes(aspects_str: Optional[str], keep_raw: bool = False) -> Dict:
    """
    Normalize aspects string to thematic categories.
    
    Two-level model:
    - aspect_raw: Original extracted text
    - aspect_theme: Thematic category for analytics
    
    IMPORTANT: Returns empty values (None/empty string) for reviews without text.
    Only reviews that were actually processed by Azure should have aspects.
    
    Args:
        aspects_str: Raw aspects string from Azure
        keep_raw: If True, also return original aspect names
        
    Returns:
        Dictionary with:
            - 'themes': List of thematic categories (deduplicated)
            - 'themes_str': Semicolon-separated string of themes (None if no aspects)
            - 'raw': List of original aspect names (if keep_raw=True)
    """
    if not aspects_str or pd.isna(aspects_str):
        # No aspects = review was not processed (no text)
        # Return None/empty, not a default theme
        return {
            "themes": [],
            "themes_str": None,  # None indicates no aspects (no text)
            "raw": [] if not keep_raw else [],
        }
    
    # Check if it's an empty string or just whitespace
    aspects_str = str(aspects_str).strip()
    if not aspects_str or aspects_str.lower() in ("nan", "none", ""):
        return {
            "themes": [],
            "themes_str": None,
            "raw": [] if not keep_raw else [],
        }
    
    # Parse individual aspects (aspect_raw)
    raw_aspects = parse_aspects_string(aspects_str)
    
    if not raw_aspects:
        # Empty aspects list = no aspects extracted (but review had text)
        return {
            "themes": [],
            "themes_str": None,
            "raw": [] if not keep_raw else raw_aspects,
        }
    
    # Normalize each aspect to thematic category
    themes_set: Set[str] = set()
    for raw_aspect in raw_aspects:
        theme = get_thematic_aspect(raw_aspect)
        themes_set.add(theme)
    
    # Convert to sorted list for consistency
    themes_list = sorted(list(themes_set))
    
    # Only return themes if we have actual aspects
    if not themes_list:
        return {
            "themes": [],
            "themes_str": None,
            "raw": raw_aspects if keep_raw else [],
        }
    
    return {
        "themes": themes_list,
        "themes_str": "; ".join(themes_list),
        "raw": raw_aspects if keep_raw else [],
    }


def normalize_aspects_column(df: pd.DataFrame, 
                             aspect_col: str = "aspects", 
                             output_theme_col: str = "aspect_theme",
                             output_raw_col: Optional[str] = "aspect_raw") -> pd.DataFrame:
    """
    Normalize aspects column in a DataFrame using two-level model.
    
    Creates:
    - aspect_raw: Original extracted aspect text
    - aspect_theme: Thematic category for analytics
    
    Args:
        df: Input DataFrame
        aspect_col: Name of column containing raw aspects from Azure
        output_theme_col: Name of output column for thematic categories
        output_raw_col: Name of output column for raw aspects (None to skip)
        
    Returns:
        DataFrame with normalized aspect columns added
    """
    df = df.copy()
    
    if aspect_col not in df.columns:
        print(f"⚠️ Column '{aspect_col}' not found. Creating default thematic column.")
        df[output_theme_col] = "guest_experience"
        if output_raw_col:
            df[output_raw_col] = ""
        return df
    
    # Apply normalization
    results = df[aspect_col].apply(
        lambda x: normalize_aspects_to_themes(x, keep_raw=output_raw_col is not None)
    )
    
    # Extract thematic categories (None for reviews without text/aspects)
    df[output_theme_col] = results.apply(lambda r: r["themes_str"] if r["themes_str"] is not None else None)
    
    # Optionally store raw aspects (empty for reviews without text/aspects)
    if output_raw_col:
        df[output_raw_col] = results.apply(
            lambda r: "; ".join(r["raw"]) if r["raw"] else None
        )
    
    return df


def get_thematic_statistics(df: pd.DataFrame, theme_col: str = "aspect_theme") -> pd.DataFrame:
    """
    Get statistics on thematic category distribution.
    
    Only counts reviews with actual aspects (non-null aspect_theme).
    Reviews without text are excluded from statistics.
    
    Args:
        df: DataFrame with normalized aspects
        theme_col: Name of thematic aspects column
        
    Returns:
        DataFrame with theme counts and percentages
    """
    if theme_col not in df.columns:
        raise ValueError(f"Column '{theme_col}' not found in DataFrame")
    
    # Split semicolon-separated themes and count
    # Only process non-null values (reviews with text)
    all_themes = []
    for themes_str in df[theme_col].dropna():
        if themes_str and str(themes_str).strip():
            all_themes.extend([t.strip() for t in str(themes_str).split(";")])
    
    # Count occurrences
    from collections import Counter
    counts = Counter(all_themes)
    
    # Calculate statistics based on reviews WITH aspects (not total reviews)
    reviews_with_aspects = df[theme_col].notna().sum()
    
    stats_df = pd.DataFrame([
        {
            "theme": theme,
            "count": count,
            "percentage": (count / reviews_with_aspects) * 100 if reviews_with_aspects > 0 else 0
        }
        for theme, count in counts.most_common()
    ])
    
    return stats_df


def validate_thematic_normalization(df: pd.DataFrame, theme_col: str = "aspect_theme") -> Dict:
    """
    Validate that all thematic categories are valid.
    
    Args:
        df: DataFrame with normalized aspects
        theme_col: Name of thematic aspects column
        
    Returns:
        Dictionary with validation results:
            - 'valid': Boolean indicating if all themes are valid
            - 'invalid_themes': Set of invalid theme names found
            - 'theme_distribution': Count of reviews per theme
    """
    if theme_col not in df.columns:
        return {
            "valid": False,
            "invalid_themes": set(),
            "theme_distribution": {},
            "error": f"Column '{theme_col}' not found",
        }
    
    all_themes = set()
    for themes_str in df[theme_col].dropna():
        if themes_str:
            themes = [t.strip() for t in str(themes_str).split(";")]
            all_themes.update(themes)
    
    invalid = all_themes - THEMATIC_CATEGORIES
    
    # Check for generic categories (should not exist)
    generic_categories = {"other", "unclassified", "misc", "unknown", "others", "miscellaneous"}
    found_generic = all_themes & generic_categories
    
    # Count distribution
    from collections import Counter
    all_themes_list = []
    for themes_str in df[theme_col].dropna():
        if themes_str:
            all_themes_list.extend([t.strip() for t in str(themes_str).split(";")])
    theme_distribution = dict(Counter(all_themes_list))
    
    return {
        "valid": len(invalid) == 0 and len(found_generic) == 0,
        "invalid_themes": invalid,
        "generic_categories_found": found_generic,
        "theme_distribution": theme_distribution,
        "total_unique_themes": len(all_themes),
    }


def create_thematic_table(df: pd.DataFrame, 
                         theme_col: str = "aspect_theme",
                         year_col: Optional[str] = "year_month") -> pd.DataFrame:
    """
    Create analytical table with themes as rows and years as columns.
    
    This produces the stable table format needed for analytics:
    
    THEME | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | TOTAL
    
    Args:
        df: DataFrame with normalized aspects
        theme_col: Name of thematic aspects column
        year_col: Name of year column (optional, if None uses all data)
        
    Returns:
        Pivot table with themes as rows and years as columns
    """
    if theme_col not in df.columns:
        raise ValueError(f"Column '{theme_col}' not found in DataFrame")
    
    # Expand semicolon-separated themes into separate rows
    expanded_rows = []
    for idx, row in df.iterrows():
        themes_str = row[theme_col]
        if pd.isna(themes_str) or not themes_str:
            continue
        
        themes = [t.strip() for t in str(themes_str).split(";")]
        year = row.get(year_col, "all") if year_col and year_col in df.columns else "all"
        
        for theme in themes:
            expanded_rows.append({
                "theme": theme,
                "year": str(year)[:4] if year != "all" else "all",  # Extract year from year_month
                "count": 1,
            })
    
    if not expanded_rows:
        return pd.DataFrame(columns=["theme", "total"])
    
    expanded_df = pd.DataFrame(expanded_rows)
    
    # Create pivot table
    if year_col and year_col in df.columns:
        pivot = expanded_df.pivot_table(
            index="theme",
            columns="year",
            values="count",
            aggfunc="sum",
            fill_value=0
        )
        pivot["total"] = pivot.sum(axis=1)
    else:
        # No year column, just count by theme
        pivot = expanded_df.groupby("theme")["count"].sum().to_frame("total")
    
    # Sort by total descending
    pivot = pivot.sort_values("total", ascending=False)
    
    return pivot
