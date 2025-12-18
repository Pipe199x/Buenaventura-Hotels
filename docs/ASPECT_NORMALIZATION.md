# Aspect Normalization System

## Overview

The aspect normalization system ensures that aspect values extracted from hotel reviews are consistently categorized into canonical English names. This prevents issues with analytics, grouping, and year-over-year comparisons caused by:

- Singular vs plural forms (baño vs baños)
- Synonyms (aire vs aire acondicionado)
- Misspellings or variants (cama vs camas)
- Overlapping meanings (aseo, atención aseo)
- Future unknown aspects from new hotels

## Architecture

### Components

1. **`aspect_mappings.py`**: Single source of truth for aspect mappings
   - Defines canonical aspect categories
   - Maps raw values to canonical names
   - Handles text normalization (accents, case, etc.)

2. **`aspect_normalization.py`**: Normalization logic
   - Parses Azure Cognitive Services aspect format
   - Normalizes individual aspects
   - Validates normalization results

3. **Integration**: Automatically applied in `enrich_with_azure()` function
   - Normalizes aspects immediately after extraction
   - Creates `aspect_normalized` column
   - Preserves original `aspects` column

## Canonical Aspect Categories

All canonical aspects use English snake_case naming:

- `bathroom_cleanliness` - Bathroom, restroom, hygiene
- `air_conditioning` - AC, climate control, ventilation
- `room_quality` - Room size, space, accommodation
- `service_quality` - General service quality
- `location` - Location, area, neighborhood
- `price_value` - Price, cost, value for money
- `food_dining` - Food, restaurant, meals
- `wifi_internet` - Internet, WiFi, connectivity
- `parking` - Parking, parking lot
- `safety_security` - Safety, security
- `noise` - Noise levels, quietness
- `bed_comfort` - Bed, mattress, comfort
- `staff_attention` - Staff, reception, customer service
- `cleanliness_general` - General cleanliness
- `facilities` - Facilities, amenities, infrastructure
- `other` - Other aspects
- `unclassified` - Unknown or unmapped aspects

## Normalization Strategy

### 1. Text Normalization
- Convert to lowercase
- Remove accents (á → a, é → e, etc.)
- Strip whitespace

### 2. Matching Strategy
1. **Direct match**: Exact match after normalization
2. **Partial match**: Check if mapping key is contained in input or vice versa
3. **Fallback**: Assign to `unclassified` if no match found

### 3. Unknown Aspects Handling
- Unknown aspects are **never discarded**
- They are assigned to `unclassified` category
- Analytics continue to work without errors
- Can be reviewed and added to mappings later

## Example Mappings

```python
# Singular/Plural
"baño" → "bathroom_cleanliness"
"baños" → "bathroom_cleanliness"

# Synonyms
"aire" → "air_conditioning"
"aire acondicionado" → "air_conditioning"
"clima" → "air_conditioning"

# Variants
"cama" → "bed_comfort"
"camas" → "bed_comfort"
"colchón" → "bed_comfort"

# Overlapping meanings
"aseo" → "bathroom_cleanliness"
"atención aseo" → "bathroom_cleanliness"
```

## Usage

### Automatic Normalization

Aspects are automatically normalized during the Gold layer enrichment:

```python
from src.domain.transformations import enrich_with_azure

# Aspects are normalized automatically
df_gold = enrich_with_azure(df_silver, language="es", mode="cloud")
# df_gold now contains both 'aspects' (original) and 'aspect_normalized' columns
```

### Manual Normalization

```python
from src.domain.aspect_normalization import normalize_aspects_column

# Normalize existing DataFrame
df = normalize_aspects_column(df, aspect_col="aspects", output_col="aspect_normalized")
```

### Adding New Mappings

```python
from src.domain.aspect_mappings import add_mapping

# Add new variant mapping
add_mapping("nueva_variante", "bathroom_cleanliness")
```

### Getting Statistics

```python
from src.domain.aspect_normalization import get_aspect_statistics

# Get aspect distribution
stats = get_aspect_statistics(df, aspect_col="aspect_normalized")
print(stats)
```

### Validation

```python
from src.domain.aspect_normalization import validate_normalization

# Validate normalization
result = validate_normalization(df, aspect_col="aspect_normalized")
if not result["valid"]:
    print(f"Invalid aspects found: {result['invalid_aspects']}")
```

## Output Format

The `aspect_normalized` column contains semicolon-separated canonical aspect names:

```
"bathroom_cleanliness; air_conditioning; room_quality"
```

Multiple aspects from the same review are normalized and deduplicated.

## Benefits

1. **Stable Groupings**: Consistent aspect names across all hotels and time periods
2. **Year-over-Year Comparisons**: Historical data remains consistent
3. **Scalability**: New hotels and aspects don't break analytics
4. **Maintainability**: Single source of truth for mappings
5. **Future-Proof**: Easy to extend with new categories or languages

## Database Integration

The `aspect_normalized` column is included in the `DATA_COLS` list and will be:
- Stored in the database alongside original `aspects` column
- Available for SQL queries and Power BI dashboards
- Used for consistent analytics and reporting

## Maintenance

### Adding New Aspects

1. Add canonical name to `CANONICAL_ASPECTS` set in `aspect_mappings.py`
2. Add variant mappings to `ASPECT_MAPPINGS` dictionary
3. Test with sample data
4. Deploy

### Reviewing Unclassified Aspects

Periodically check for high counts of `unclassified` aspects:

```python
unclassified_count = df["aspect_normalized"].str.contains("unclassified", na=False).sum()
if unclassified_count > threshold:
    # Review and add mappings
    pass
```

## Design Principles

1. **Single Source of Truth**: All mappings in one place
2. **Never Discard Data**: Unknown aspects → `unclassified`
3. **Deterministic**: Same input always produces same output
4. **Extensible**: Easy to add new mappings
5. **Clean Architecture**: Separation of concerns, domain logic isolated

