# Aspect Normalization System - Implementation Summary

## Problem Statement

The dataset contained non-normalized aspect values from Azure Cognitive Services, causing:
- Inconsistent groupings in analytical tables
- Broken year-over-year comparisons
- Scalability issues when new hotels/data sources are added
- Manual fixes required in visualizations

## Solution Overview

Implemented a robust, scalable aspect normalization system that:
1. Maps raw aspect values to canonical English names
2. Handles singular/plural, synonyms, variants, and accents
3. Never discards unknown aspects (assigns to `unclassified`)
4. Provides a single source of truth for mappings
5. Integrates seamlessly into the existing ETL pipeline

## Implementation Details

### 1. Architecture

```
src/domain/
├── aspect_mappings.py          # Single source of truth for mappings
├── aspect_normalization.py     # Normalization logic
└── transformations.py          # Integration point (enrich_with_azure)
```

### 2. Key Components

#### `aspect_mappings.py`
- **CANONICAL_ASPECTS**: Set of 17 canonical English aspect categories
- **ASPECT_MAPPINGS**: Dictionary mapping raw values → canonical names
- **get_canonical_aspect()**: Core mapping function with text normalization
- **add_mapping()**: Dynamic mapping addition for extensibility

#### `aspect_normalization.py`
- **parse_aspects_string()**: Extracts individual aspects from Azure format
- **normalize_aspects()**: Normalizes a single aspects string
- **normalize_aspects_column()**: DataFrame-level normalization
- **validate_normalization()**: Validation and quality checks
- **get_aspect_statistics()**: Analytics helper

### 3. Normalization Strategy

**Text Normalization:**
- Lowercase conversion
- Accent removal (á → a, é → e, etc.)
- Whitespace trimming

**Matching Strategy:**
1. Direct match (exact after normalization)
2. Partial match (substring matching)
3. Fallback to `unclassified`

**Unknown Aspects:**
- Never discarded
- Assigned to `unclassified` category
- Analytics continue to work
- Can be reviewed and mapped later

### 4. Integration

Aspect normalization is automatically applied in the `enrich_with_azure()` function:

```python
# In domain/transformations.py
def enrich_with_azure(...):
    # ... Azure enrichment ...
    
    # Normalize aspects to canonical English names
    df = normalize_aspects_column(df, aspect_col="aspects", output_col="aspect_normalized")
    
    # Validate and log results
    validation = validate_normalization(df, aspect_col="aspect_normalized")
    # ...
```

### 5. Output

The system creates a new column `aspect_normalized` containing:
- Semicolon-separated canonical aspect names
- Example: `"bathroom_cleanliness; air_conditioning; room_quality"`
- Original `aspects` column is preserved

## Example Mappings

| Raw Input | Canonical Output |
|-----------|------------------|
| `"baño"` | `"bathroom_cleanliness"` |
| `"baños"` | `"bathroom_cleanliness"` |
| `"aire"` | `"air_conditioning"` |
| `"aire acondicionado"` | `"air_conditioning"` |
| `"cama"` | `"bed_comfort"` |
| `"camas"` | `"bed_comfort"` |
| `"aseo"` | `"bathroom_cleanliness"` |
| `"atención aseo"` | `"bathroom_cleanliness"` |
| `"unknown_xyz"` | `"unclassified"` |

## Canonical Aspect Categories

1. `bathroom_cleanliness` - Bathroom, restroom, hygiene
2. `air_conditioning` - AC, climate control
3. `room_quality` - Room size, space
4. `service_quality` - General service
5. `location` - Location, area
6. `price_value` - Price, value for money
7. `food_dining` - Food, restaurant
8. `wifi_internet` - Internet, WiFi
9. `parking` - Parking facilities
10. `safety_security` - Safety, security
11. `noise` - Noise levels
12. `bed_comfort` - Bed, mattress
13. `staff_attention` - Staff, reception
14. `cleanliness_general` - General cleanliness
15. `facilities` - Facilities, amenities
16. `other` - Other aspects
17. `unclassified` - Unknown/unmapped

## Benefits

### 1. Stable Groupings
- Consistent aspect names across all hotels
- No more "baño" vs "baños" splitting

### 2. Year-over-Year Comparisons
- Historical data remains consistent
- New hotels don't break existing analytics

### 3. Scalability
- New hotels automatically handled
- New aspects can be added via mappings
- No code changes needed for new variants

### 4. Maintainability
- Single source of truth (`aspect_mappings.py`)
- Easy to audit and update
- Clear separation of concerns

### 5. Future-Proof
- Ready for new languages
- Extensible mapping system
- Validation and monitoring built-in

## Usage Examples

### Automatic (Recommended)
Aspects are normalized automatically during Gold layer processing:
```python
df_gold = enrich_with_azure(df_silver, language="es", mode="cloud")
# df_gold now has 'aspect_normalized' column
```

### Manual Normalization
```python
from src.domain.aspect_normalization import normalize_aspects_column

df = normalize_aspects_column(df, aspect_col="aspects", output_col="aspect_normalized")
```

### Adding New Mappings
```python
from src.domain.aspect_mappings import add_mapping

add_mapping("nueva_variante", "bathroom_cleanliness")
```

### Getting Statistics
```python
from src.domain.aspect_normalization import get_aspect_statistics

stats = get_aspect_statistics(df, aspect_col="aspect_normalized")
```

## Database Integration

The `aspect_normalized` column is:
- Included in `DATA_COLS` for database insertion
- Stored alongside original `aspects` column
- Available for SQL queries and Power BI
- Ready for analytical views

## Maintenance

### Adding New Aspects
1. Add canonical name to `CANONICAL_ASPECTS` set
2. Add variant mappings to `ASPECT_MAPPINGS` dictionary
3. Test with sample data
4. Deploy

### Monitoring Unclassified
```python
unclassified_count = df["aspect_normalized"].str.contains("unclassified", na=False).sum()
# Review if count is high
```

## Design Principles

1. **Single Source of Truth**: All mappings in `aspect_mappings.py`
2. **Never Discard Data**: Unknown → `unclassified`
3. **Deterministic**: Same input → same output
4. **Extensible**: Easy to add mappings
5. **Clean Architecture**: Domain logic isolated

## Testing

A test script is provided at `src/domain/test_aspect_normalization.py` demonstrating:
- Aspect string parsing
- Canonical mapping
- Full normalization
- DataFrame operations
- Validation

## Files Modified/Created

### New Files
- `src/domain/aspect_mappings.py` - Mapping definitions
- `src/domain/aspect_normalization.py` - Normalization logic
- `src/domain/test_aspect_normalization.py` - Test script
- `docs/ASPECT_NORMALIZATION.md` - Full documentation
- `docs/ASPECT_NORMALIZATION_SUMMARY.md` - This file

### Modified Files
- `src/domain/transformations.py` - Added normalization integration
- `src/domain/__init__.py` - Exported new functions
- `src/nashor_to_supabase.py` - Added `aspect_normalized` to DATA_COLS

## Next Steps

1. **Deploy**: The system is ready for production use
2. **Monitor**: Track `unclassified` counts to identify new aspects
3. **Iterate**: Add mappings as new variants are discovered
4. **Analytics**: Update Power BI dashboards to use `aspect_normalized`

## Conclusion

The aspect normalization system provides a robust, scalable solution that:
- Fixes current analytics issues
- Prevents future problems
- Maintains data integrity
- Follows clean architecture principles
- Is easy to maintain and extend

The system is production-ready and will automatically normalize aspects for all new data processed through the Gold layer.

