# Thematic Aspect Normalization System

## Overview

The thematic aspect normalization system ensures that all aspect values extracted from hotel reviews are consistently categorized into **meaningful thematic categories** aligned with analytical reporting needs. 

**Key Principle**: NO generic categories (other, unclassified, misc) are allowed. Every aspect must map to a meaningful thematic category.

## Two-Level Model

The system implements a two-level semantic structure:

1. **`aspect_raw`**: Original extracted text from Azure Cognitive Services
   - Example: `"baños"`, `"aseo"`, `"atención aseo"`

2. **`aspect_theme`**: Normalized thematic category for analytics
   - Example: `"bathrooms_cleanliness"`

All analytics and reporting use `aspect_theme` for consistent grouping and year-over-year comparisons.

## Thematic Categories

The system defines **15 meaningful thematic categories** aligned with hotel analytics:

| Theme | Description | Example Raw Aspects |
|-------|-------------|-------------------|
| `service_quality` | General service quality | servicio, calidad de servicio |
| `staff_attention` | Staff treatment and attention | personal, recepción, trato, amabilidad |
| `rooms_accommodation` | Room quality and space | habitación, cuarto, espacio, vista |
| `bathrooms_cleanliness` | Bathroom and hygiene | baño, aseo, sanitarios, ducha |
| `food_dining` | Food and restaurant | comida, restaurante, desayuno, menú |
| `maintenance_facilities` | Maintenance and repairs | mantenimiento, reparación, funcionamiento |
| `infrastructure_amenities` | Infrastructure and amenities | instalaciones, piscina, gimnasio, aire acondicionado |
| `cleanliness_general` | General cleanliness | limpieza general, higiene, orden |
| `pricing_value` | Pricing and value | precio, costo, tarifa, relación calidad precio |
| `connectivity_technology` | Connectivity and technology | wifi, internet, conexión, televisión |
| `guest_experience` | Overall guest experience | experiencia, estancia, satisfacción, ambiente |
| `location_surroundings` | Location and surroundings | ubicación, zona, barrio, cercanía |
| `safety_security` | Safety and security | seguridad, protección, vigilancia |
| `noise_quietness` | Noise levels | ruido, tranquilidad, silencio |
| `comfort_furnishings` | Comfort and furnishings | cama, colchón, comodidad, mobiliario |

## Normalization Strategy

### Multi-Step Matching Process

1. **Direct Match**: Exact match after text normalization (lowercase, accent removal)
2. **Partial Match**: Substring matching (variant contained in input or vice versa)
3. **Semantic Keyword Matching**: Match based on semantic keywords
4. **Intelligent Fallback**: Assign to most appropriate high-level theme based on content analysis
5. **Final Fallback**: Assign to `guest_experience` (meaningful, not generic)

### Text Normalization

- Convert to lowercase
- Remove accents (á → a, é → e, ñ → n, etc.)
- Strip whitespace

### Example Mappings

```
Raw Input                    →  Thematic Category
─────────────────────────────────────────────────────
"baño", "baños"             →  bathrooms_cleanliness
"aseo", "atención aseo"     →  bathrooms_cleanliness
"aire", "aire acondicionado" →  infrastructure_amenities
"servicio"                  →  service_quality
"personal", "trato"         →  staff_attention
"habitación", "cuarto"      →  rooms_accommodation
"precio", "costo"           →  pricing_value
"comida", "restaurante"      →  food_dining
"wifi", "internet"          →  connectivity_technology
"ubicación", "zona"         →  location_surroundings
```

## No Generic Categories Policy

**Critical Constraint**: The system NEVER creates generic categories like:
- ❌ `other`
- ❌ `unclassified`
- ❌ `misc`
- ❌ `unknown`
- ❌ `others`
- ❌ `miscellaneous`

Instead, unmapped aspects are assigned to meaningful high-level themes:
- ✅ `guest_experience` (for general/ambiguous aspects)
- ✅ `service_quality` (for service-related aspects)
- ✅ `infrastructure_amenities` (for facility-related aspects)

This ensures all analytics remain interpretable and meaningful.

## Analytics-Safe Output

The `aspect_theme` column produces stable analytical tables:

```
THEME                    | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | TOTAL
─────────────────────────────────────────────────────────────────────────
bathrooms_cleanliness    |  45  |  52  |  48  |  61  |  55  |  58  |  319
service_quality          |  38  |  42  |  40  |  45  |  43  |  47  |  255
staff_attention          |  32  |  35  |  33  |  38  |  36  |  39  |  213
rooms_accommodation      |  28  |  31  |  29  |  34  |  32  |  35  |  189
...
```

### Benefits

- **No Fragmentation**: All variants map to same theme
- **Stable Groupings**: Consistent across years
- **Historical Comparability**: Year-over-year comparisons work correctly
- **No Duplicates**: Single theme per aspect type

## Usage

### Automatic Normalization

Aspects are automatically normalized during Gold layer enrichment:

```python
from src.domain.transformations import enrich_with_azure

# Aspects are normalized automatically
df_gold = enrich_with_azure(df_silver, language="es", mode="cloud")
# df_gold now contains:
# - 'aspects' (original Azure format)
# - 'aspect_raw' (extracted raw text)
# - 'aspect_theme' (thematic category for analytics)
```

### Manual Normalization

```python
from src.domain.aspect_normalization import normalize_aspects_column

# Normalize existing DataFrame
df = normalize_aspects_column(
    df, 
    aspect_col="aspects", 
    output_theme_col="aspect_theme",
    output_raw_col="aspect_raw"
)
```

### Creating Analytical Tables

```python
from src.domain.aspect_normalization import create_thematic_table

# Create year-by-year thematic table
thematic_table = create_thematic_table(
    df, 
    theme_col="aspect_theme",
    year_col="year_month"
)
# Returns pivot table: themes as rows, years as columns
```

### Getting Statistics

```python
from src.domain.aspect_normalization import get_thematic_statistics

# Get thematic distribution
stats = get_thematic_statistics(df, theme_col="aspect_theme")
print(stats)
```

### Adding New Mappings

```python
from src.domain.aspect_mappings import add_thematic_mapping

# Add new variant mapping
add_thematic_mapping("nueva_variante", "bathrooms_cleanliness")
```

### Validation

```python
from src.domain.aspect_normalization import validate_thematic_normalization

# Validate normalization
result = validate_thematic_normalization(df, theme_col="aspect_theme")
if not result["valid"]:
    print(f"Invalid themes: {result['invalid_themes']}")
if result.get("generic_categories_found"):
    print(f"ERROR: Generic categories found: {result['generic_categories_found']}")
```

## Database Integration

The system creates two columns:
- `aspect_raw`: Original extracted aspect text (for reference)
- `aspect_theme`: Thematic category (for analytics)

Both columns are:
- Included in `DATA_COLS` for database insertion
- Stored in the database
- Available for SQL queries and Power BI dashboards

## Scalability

### Adding New Hotels
- New hotels automatically use existing thematic mappings
- No code changes needed
- Analytics remain consistent

### Adding New Aspects
1. Add variant to `THEMATIC_MAPPINGS` in `aspect_mappings.py`
2. Map to appropriate existing theme
3. Or create new theme in `THEMATIC_CATEGORIES` (if truly new dimension)

### Adding New Languages
- Extend `THEMATIC_MAPPINGS` with language-specific variants
- Same themes, different raw values
- Analytics remain consistent across languages

## Design Principles

1. **Meaningful Categories Only**: No generic catch-all categories
2. **Single Source of Truth**: All mappings in `aspect_mappings.py`
3. **Two-Level Model**: Raw text + thematic category
4. **Intelligent Fallback**: Always assigns to meaningful theme
5. **Analytics-First**: Designed for stable analytical tables
6. **Extensible**: Easy to add variants and themes

## Maintenance

### Reviewing Unmapped Aspects

Periodically check for aspects that fall back to high-level themes:

```python
# Find reviews with guest_experience theme (might indicate unmapped aspects)
guest_exp_reviews = df[df["aspect_theme"].str.contains("guest_experience", na=False)]
# Review aspect_raw values to identify new mappings needed
```

### Adding New Themes

Only add new themes when they represent a truly new analytical dimension:

1. Add theme to `THEMATIC_CATEGORIES` set
2. Add variant mappings to `THEMATIC_MAPPINGS`
3. Update semantic keywords if needed
4. Test with sample data
5. Deploy

## Example: Complete Flow

```
Azure Output:
"baños (negative): sucio, mal olor | servicio (positive): amable"

↓ Parse

aspect_raw:
"baños; servicio"

↓ Normalize

aspect_theme:
"bathrooms_cleanliness; service_quality"

↓ Analytics

Thematic Table:
bathrooms_cleanliness | 2020 | 2021 | ... | TOTAL
service_quality       | 2020 | 2021 | ... | TOTAL
```

## Benefits

1. **Interpretable Analytics**: All categories are meaningful business dimensions
2. **Stable Groupings**: No fragmentation across variants
3. **Year-over-Year Consistency**: Historical comparisons work correctly
4. **Scalable**: Handles new hotels and aspects automatically
5. **Maintainable**: Single source of truth, easy to extend
6. **Future-Proof**: Ready for multilingual expansion

## Files

- `src/domain/aspect_mappings.py`: Thematic mappings (single source of truth)
- `src/domain/aspect_normalization.py`: Normalization logic
- `src/domain/transformations.py`: Integration point
- `src/nashor_to_supabase.py`: Database column definitions

## Summary

The thematic aspect normalization system ensures:
- ✅ All aspects map to meaningful thematic categories
- ✅ NO generic categories (other, unclassified, etc.)
- ✅ Stable analytical tables for year-over-year comparisons
- ✅ Scalable for new hotels and aspects
- ✅ Clean, maintainable, extensible design

