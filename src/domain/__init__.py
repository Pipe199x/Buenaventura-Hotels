"""Domain layer for business logic and data transformations."""

from .transformations import (
    make_silver,
    enrich_with_azure,
    normalize_dataframe,
    union_and_deduplicate,
)
from .aspect_normalization import (
    normalize_aspects_to_themes,
    normalize_aspects_column,
    parse_aspects_string,
    get_thematic_statistics,
    validate_thematic_normalization,
    create_thematic_table,
)
from .aspect_mappings import (
    get_thematic_aspect,
    add_thematic_mapping,
    get_all_variants,
    get_thematic_statistics as get_mapping_stats,
    THEMATIC_CATEGORIES,
)

__all__ = [
    "make_silver",
    "enrich_with_azure",
    "normalize_dataframe",
    "union_and_deduplicate",
    "normalize_aspects_to_themes",
    "normalize_aspects_column",
    "parse_aspects_string",
    "get_thematic_statistics",
    "validate_thematic_normalization",
    "create_thematic_table",
    "get_thematic_aspect",
    "add_thematic_mapping",
    "get_all_variants",
    "get_mapping_stats",
    "THEMATIC_CATEGORIES",
]

