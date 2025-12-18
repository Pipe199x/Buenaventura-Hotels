"""
Example script to create thematic tables matching manual dictionary approach.

This script demonstrates how to create clean thematic tables where:
- Each theme appears on its own row (not combinations)
- Counts are aggregated by individual themes
- Matches the manual dictionary approach
"""

import pandas as pd
from pathlib import Path
from domain.aspect_normalization import create_thematic_table

# Load your Gold parquet file
# Adjust path as needed
parquet_path = Path("data/gold/cordillera_GOLD.parquet")  # or load from Azure

# Load data
df = pd.read_parquet(parquet_path)

# Optionally filter by hotel
# df = df[df['hotel_id'] == 'cordillera']

# Create thematic table (explodes combinations automatically)
table = create_thematic_table(
    df,
    theme_col="aspect_theme",
    year_col="year_month",  # or "publishedAtDate"
    sentiment_col="sentiment_label",  # Optional: filter by sentiment
    sentiment_filter=None  # "positive", "negative", or None for all
)

print("=" * 80)
print("THEMATIC TABLE - Individual Themes (Exploded Combinations)")
print("=" * 80)
print(table)

# For positive sentiment only
print("\n" + "=" * 80)
print("POSITIVE SENTIMENT THEMES")
print("=" * 80)
table_positive = create_thematic_table(
    df,
    theme_col="aspect_theme",
    year_col="year_month",
    sentiment_col="sentiment_label",
    sentiment_filter="positive"
)
print(table_positive)

# For negative sentiment only
print("\n" + "=" * 80)
print("NEGATIVE SENTIMENT THEMES")
print("=" * 80)
table_negative = create_thematic_table(
    df,
    theme_col="aspect_theme",
    year_col="year_month",
    sentiment_col="sentiment_label",
    sentiment_filter="negative"
)
print(table_negative)

# Export to Excel
output_path = Path("thematic_tables.xlsx")
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    table.to_excel(writer, sheet_name="All_Sentiments")
    table_positive.to_excel(writer, sheet_name="Positive")
    table_negative.to_excel(writer, sheet_name="Negative")

print(f"\n✅ Tables exported to: {output_path}")

