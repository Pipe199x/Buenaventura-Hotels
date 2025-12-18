# Creating Thematic Tables - Matching Manual Dictionary Approach

## Problem

The Python output shows **theme combinations** (e.g., "bathrooms_cleanliness; service_quality"), creating many rows. Your manual table uses **single themes per row**, which is cleaner for analytics.

## Solution

The `create_thematic_table()` function **automatically explodes combinations**, so each theme appears on its own row. This matches your manual approach.

## Key Difference

### Your Manual Approach
- "Habitaciones" = 33 total
- "Atención" = 14 total
- Each theme counted individually

### Python Output (Before Fix)
- "bathrooms_cleanliness; service_quality" = 1 (combination)
- "bathrooms_cleanliness" = 2 (individual)
- Creates many rows with combinations

### Python Output (After Fix - Exploded)
- "bathrooms_cleanliness" = 15 (all occurrences, including in combinations)
- "service_quality" = 12 (all occurrences, including in combinations)
- Each theme on its own row (matches your manual approach)

## How It Works

When a review has multiple themes like:
```
aspect_theme: "bathrooms_cleanliness; service_quality; rooms_accommodation"
```

The function **explodes** it into:
- 1 count for "bathrooms_cleanliness"
- 1 count for "service_quality"  
- 1 count for "rooms_accommodation"

So if you have 10 reviews with this combination:
- "bathrooms_cleanliness" gets +10
- "service_quality" gets +10
- "rooms_accommodation" gets +10

## Usage Example

```python
import pandas as pd
from src.domain.aspect_normalization import create_thematic_table

# Load your data
df = pd.read_parquet("data/gold/cordillera_GOLD.parquet")

# Create table (automatically explodes combinations)
table = create_thematic_table(
    df,
    theme_col="aspect_theme",
    year_col="year_month",
    sentiment_col="sentiment_label",  # Optional
    sentiment_filter="positive"  # Optional: "positive", "negative", or None
)

print(table)
```

## Output Format

The output will match your manual table structure:

```
THEME                          | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | TOTAL
─────────────────────────────────────────────────────────────────────────────
rooms_accommodation            |  4   |  8   |  7   |  9   |  3   |  2   |  33
service_quality                |  2   |  1   |  4   |  3   |  7   |  3   |  20
staff_attention                |  1   |  2   |  5   |  4   |  1   |  1   |  14
bathrooms_cleanliness          |  1   |  1   |  0   |  4   |  3   |  1   |  10
...
TOTAL                          | 12   | 23   | 32   | 46   | 24   | 38   | 175
```

## Filtering by Sentiment

To match your "Temáticas positivas" table:

```python
# Positive sentiment only
table_positive = create_thematic_table(
    df,
    theme_col="aspect_theme",
    year_col="year_month",
    sentiment_col="sentiment_label",
    sentiment_filter="positive"
)
```

## Theme Name Mapping

Your manual table uses Spanish names, but the system uses English themes. You can create a mapping:

```python
# Theme name mapping (Spanish → English)
THEME_NAMES = {
    "rooms_accommodation": "Habitaciones",
    "service_quality": "Servicio",
    "staff_attention": "Atención",
    "bathrooms_cleanliness": "Aseo / Baños",
    "food_dining": "Comida / Restaurante",
    "guest_experience": "Experiencia",
    "infrastructure_amenities": "Instalaciones",
    "pricing_value": "Precio",
    "connectivity_technology": "Tecnología",
    "location_surroundings": "Ubicación",
    # ... etc
}

# Apply mapping
table.index = table.index.map(lambda x: THEME_NAMES.get(x, x))
```

## SQL Query Alternative

You can also create the table directly in SQL:

```sql
-- Explode themes and count by year
WITH exploded_themes AS (
    SELECT 
        reviewId,
        EXTRACT(YEAR FROM "publishedAtDate") as year,
        TRIM(unnest(string_to_array(aspect_theme, ';'))) as theme,
        sentiment_label
    FROM hotels_gold
    WHERE hotel_name = 'cordillera'
      AND aspect_theme IS NOT NULL
      AND EXTRACT(YEAR FROM "publishedAtDate") BETWEEN 2020 AND 2025
)
SELECT 
    theme,
    COUNT(*) FILTER (WHERE year = 2020) as "2020",
    COUNT(*) FILTER (WHERE year = 2021) as "2021",
    COUNT(*) FILTER (WHERE year = 2022) as "2022",
    COUNT(*) FILTER (WHERE year = 2023) as "2023",
    COUNT(*) FILTER (WHERE year = 2024) as "2024",
    COUNT(*) FILTER (WHERE year = 2025) as "2025",
    COUNT(*) as total
FROM exploded_themes
GROUP BY theme
ORDER BY total DESC;
```

## Why This Approach Works

1. **Matches Manual Dictionary**: Each theme on its own row
2. **Accurate Counts**: All occurrences counted (including in combinations)
3. **Clean Analytics**: No fragmented combinations
4. **Year-over-Year**: Stable theme names for comparisons
5. **Scalable**: Works for any number of themes

## Summary

✅ The `create_thematic_table()` function **automatically explodes combinations**
✅ Each theme appears on its own row (matches your manual approach)
✅ Counts are accurate (includes all occurrences)
✅ You can filter by sentiment (positive/negative)
✅ Output format matches your manual table structure

The key is that combinations are **exploded** so "bathrooms_cleanliness; service_quality" contributes to both themes individually, not as a separate row.

