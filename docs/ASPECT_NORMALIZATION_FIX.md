# Aspect Normalization Fix - Handling Reviews Without Text

## Problem

The aspect normalization was being applied to **all reviews** (1193 total), including reviews **without text**. This caused:

- Reviews without text were assigned default themes (e.g., "guest_experience")
- Aspect counts were inflated (1193 instead of 733)
- SQL queries filtering by aspect polarity returned incorrect totals
- Analytics showed aspects for reviews that had no text to analyze

## Root Cause

1. Azure Cognitive Services only processes reviews **with text**
2. Reviews without text don't get sentiment analysis or aspect extraction
3. The normalization function was assigning default themes to empty/null aspects
4. This made it appear that all reviews had aspects

## Solution

### Changes Made

1. **Updated `normalize_aspects_to_themes()`**:
   - Returns `None` (not a default theme) for reviews without text
   - Only assigns themes when aspects actually exist
   - Empty/null aspects → `None` (not "guest_experience")

2. **Updated `normalize_aspects_column()`**:
   - Sets `aspect_theme = None` for reviews without text
   - Sets `aspect_raw = None` for reviews without text
   - Only reviews with actual Azure-processed aspects get themes

3. **Updated statistics and validation**:
   - Only counts reviews with non-null `aspect_theme`
   - Reports both: reviews with aspects vs. reviews without text

### Behavior After Fix

**Before**:
```
Total reviews: 1193
All reviews have aspect_theme (default: "guest_experience")
Aspect counts: 1193 (incorrect)
```

**After**:
```
Total reviews: 1193
Reviews with text: 733 → aspect_theme = "bathrooms_cleanliness; service_quality" (actual themes)
Reviews without text: 460 → aspect_theme = NULL
Aspect counts: 733 (correct)
```

## SQL Query Impact

### Before Fix
```sql
SELECT COUNT(*) 
FROM hotels_gold 
WHERE hotel_name = 'cordillera' 
  AND aspect_theme IS NOT NULL;
-- Returns: 1193 (incorrect - includes reviews without text)
```

### After Fix
```sql
SELECT COUNT(*) 
FROM hotels_gold 
WHERE hotel_name = 'cordillera' 
  AND aspect_theme IS NOT NULL;
-- Returns: 733 (correct - only reviews with text)
```

### Filtering by Aspect Polarity

Your query will now work correctly:
```sql
SELECT
    hotel_name,
    review_year,
    aspect_theme AS aspect_clean,
    text_used AS review_text
FROM hotels_gold
WHERE hotel_name = 'cordillera'
  AND review_year BETWEEN 2020 AND 2025
  AND aspect_polarity = 'positive'  -- or 'negative'
  AND aspect_theme IS NOT NULL      -- Only reviews with text
ORDER BY publishedAtDate, aspect_theme;
-- Returns: 733 reviews (only those with text and aspects)
```

## Data Structure

### Reviews WITH Text
```
reviewId: "abc123"
text_used: "El baño estaba sucio..."
aspects: "baño (negative): sucio, mal olor"
aspect_raw: "baño"
aspect_theme: "bathrooms_cleanliness"
sentiment_label: "negative"
```

### Reviews WITHOUT Text
```
reviewId: "xyz789"
text_used: NULL (or empty)
aspects: NULL
aspect_raw: NULL
aspect_theme: NULL
sentiment_label: NULL
```

## Verification

After reprocessing, verify with:

```sql
-- Count reviews with aspects (should match Azure-processed count)
SELECT 
    hotel_name,
    COUNT(*) as total_reviews,
    COUNT(aspect_theme) as reviews_with_aspects,
    COUNT(*) - COUNT(aspect_theme) as reviews_without_text
FROM hotels_gold
WHERE hotel_name = 'cordillera'
GROUP BY hotel_name;

-- Expected:
-- total_reviews: 1193
-- reviews_with_aspects: 733
-- reviews_without_text: 460
```

## Migration Notes

### For Existing Data

If you've already loaded data with the old normalization:
1. Reprocess Gold layer: `python -m src.gold_build --hotel all --mode cloud`
2. Reload to database: `python -m src.nashor_to_supabase --truncate`

### For New Data

The fix is automatic - new data will have:
- `aspect_theme = NULL` for reviews without text
- `aspect_theme = "theme1; theme2"` for reviews with aspects

## Benefits

1. **Accurate Counts**: Aspect statistics only include reviews with text
2. **Correct Filtering**: SQL queries return expected results
3. **Data Integrity**: NULL clearly indicates "no text to analyze"
4. **Analytics Accuracy**: Dashboards show correct aspect distributions

## Summary

- ✅ Reviews without text → `aspect_theme = NULL` (not a default theme)
- ✅ Only reviews with text get aspect normalization
- ✅ Aspect counts now match Azure-processed reviews (733, not 1193)
- ✅ SQL queries filter correctly by aspect polarity

