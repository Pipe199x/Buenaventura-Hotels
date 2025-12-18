"""
CORRECTED Example: Create thematic table matching manual dictionary approach.

This script shows how to create a clean table with:
- Single themes per row (not combinations)
- Counts aggregated by individual themes
- Matches your manual dictionary table format
"""

import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.domain.aspect_normalization import create_thematic_table

# ============================================================================
# STEP 1: Load your Gold parquet file
# ============================================================================
parquet_path = Path("data/gold/cordillera_GOLD.parquet")  # Adjust path as needed

if not parquet_path.exists():
    print(f"❌ File not found: {parquet_path}")
    print("💡 Make sure you've run the Gold layer processing first.")
    sys.exit(1)

df = pd.read_parquet(parquet_path)

print(f"📊 Loaded {len(df)} reviews from {parquet_path}")
print(f"📅 Date range: {df['publishedAtDate'].min()} to {df['publishedAtDate'].max()}")

# ============================================================================
# STEP 2: Filter by hotel (if needed)
# ============================================================================
# df = df[df['hotel_id'] == 'cordillera']

# ============================================================================
# STEP 3: Create thematic table (ALL SENTIMENTS)
# ============================================================================
print("\n" + "=" * 80)
print("CREATING THEMATIC TABLE - ALL SENTIMENTS")
print("=" * 80)

table_all = create_thematic_table(
    df,
    theme_col="aspect_theme",
    year_col="publishedAtDate",  # or "year_month"
    sentiment_col=None,  # No sentiment filter
    sentiment_filter=None
)

print("\n📋 THEMATIC TABLE - Individual Themes (Exploded Combinations)")
print("-" * 80)
print(table_all)

# ============================================================================
# STEP 4: Create thematic table (POSITIVE SENTIMENT ONLY)
# ============================================================================
print("\n" + "=" * 80)
print("CREATING THEMATIC TABLE - POSITIVE SENTIMENT")
print("=" * 80)

table_positive = create_thematic_table(
    df,
    theme_col="aspect_theme",
    year_col="publishedAtDate",
    sentiment_col="sentiment_label",
    sentiment_filter="positive"
)

print("\n📋 POSITIVE SENTIMENT THEMES")
print("-" * 80)
print(table_positive)

# ============================================================================
# STEP 5: Create thematic table (NEGATIVE SENTIMENT ONLY)
# ============================================================================
print("\n" + "=" * 80)
print("CREATING THEMATIC TABLE - NEGATIVE SENTIMENT")
print("=" * 80)

table_negative = create_thematic_table(
    df,
    theme_col="aspect_theme",
    year_col="publishedAtDate",
    sentiment_col="sentiment_label",
    sentiment_filter="negative"
)

print("\n📋 NEGATIVE SENTIMENT THEMES")
print("-" * 80)
print(table_negative)

# ============================================================================
# STEP 6: Optional - Map English themes to Spanish names
# ============================================================================
THEME_NAMES_SPANISH = {
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
    "cleanliness_general": "Limpieza",
    "comfort_furnishings": "Camas / Colchones",
    "maintenance_facilities": "Mantenimiento",
    "safety_security": "Seguridad",
    "noise_quietness": "Ruido / Tranquilidad",
}

def apply_spanish_names(table):
    """Apply Spanish theme names to table index."""
    table_mapped = table.copy()
    table_mapped.index = table_mapped.index.map(
        lambda x: THEME_NAMES_SPANISH.get(x, x)
    )
    return table_mapped

print("\n" + "=" * 80)
print("THEMATIC TABLE WITH SPANISH NAMES")
print("=" * 80)
table_spanish = apply_spanish_names(table_all)
print(table_spanish)

# ============================================================================
# STEP 7: Export to Excel
# ============================================================================
output_path = Path("thematic_tables_cordillera.xlsx")
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    table_all.to_excel(writer, sheet_name="All_Sentiments")
    table_positive.to_excel(writer, sheet_name="Positive")
    table_negative.to_excel(writer, sheet_name="Negative")
    table_spanish.to_excel(writer, sheet_name="All_Spanish")

print(f"\n✅ Tables exported to: {output_path}")

# ============================================================================
# STEP 8: Compare with your manual counts
# ============================================================================
print("\n" + "=" * 80)
print("COMPARISON WITH MANUAL DICTIONARY")
print("=" * 80)

# Your manual counts (from your table)
manual_counts = {
    "Habitaciones": 33,
    "Hotel": 20,  # This might map to "infrastructure_amenities" or "guest_experience"
    "Atención": 14,
    "Aseo": 10,
    "Servicio": 9,
    "Personal / empleados": 9,  # This is "staff_attention"
    "Baños": 6,
    "Lugar / sitio": 6,  # This is "location_surroundings"
    "Camas / colchones": 7,  # This is "comfort_furnishings"
    "Instalaciones": 4,
    "Experiencia": 5,
    "Teléfono / teléfonos": 4,  # This might be "connectivity_technology"
    "Pintura": 3,  # This might be "maintenance_facilities"
    "Desayuno": 3,  # This is "food_dining"
    "Olores / olor": 3,  # This might be "cleanliness_general" or "guest_experience"
    "Aire acondicionado": 3,  # This is "infrastructure_amenities"
    "Paredes": 1,  # This might be "maintenance_facilities"
    "Sábanas": 1,  # This might be "comfort_furnishings"
    "TV": 1,  # This is "connectivity_technology"
    "Citófonos": 1,  # This might be "infrastructure_amenities"
    "Facturación electrónica": 1,  # This might be "service_quality"
}

# Reverse mapping (Spanish → English)
spanish_to_english = {
    "Habitaciones": "rooms_accommodation",
    "Atención": "staff_attention",
    "Aseo": "bathrooms_cleanliness",
    "Servicio": "service_quality",
    "Personal / empleados": "staff_attention",
    "Baños": "bathrooms_cleanliness",
    "Lugar / sitio": "location_surroundings",
    "Camas / colchones": "comfort_furnishings",
    "Instalaciones": "infrastructure_amenities",
    "Experiencia": "guest_experience",
    "Teléfono / teléfonos": "connectivity_technology",
    "Desayuno": "food_dining",
    "Aire acondicionado": "infrastructure_amenities",
    "TV": "connectivity_technology",
}

print("\n📊 Comparison:")
print(f"{'Theme (Spanish)':<30} {'Manual':<10} {'Python':<10} {'Diff':<10}")
print("-" * 60)

for spanish_name, manual_count in manual_counts.items():
    english_theme = spanish_to_english.get(spanish_name)
    if english_theme and english_theme in table_all.index:
        python_count = int(table_all.loc[english_theme, "total"])
        diff = python_count - manual_count
        status = "✅" if diff == 0 else "⚠️"
        print(f"{spanish_name:<30} {manual_count:<10} {python_count:<10} {diff:+d} {status}")
    else:
        print(f"{spanish_name:<30} {manual_count:<10} {'N/A':<10} {'?':<10} ⚠️ (not mapped)")

print("\n💡 Note: Some manual themes might map to multiple English themes or need custom mapping.")

