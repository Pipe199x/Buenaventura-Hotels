# Pipeline Execution Guide

Complete step-by-step guide to run the data pipeline from scratch.

## Pipeline Overview

The pipeline consists of three main stages:

1. **Bronze → Silver**: Clean and transform raw Excel data
2. **Silver → Gold**: Enrich with Azure Cognitive Services (sentiment, aspects)
3. **Gold → Database**: Load enriched data into Supabase

## Prerequisites

### 1. Environment Setup

Create a `.env` file in the project root with:

```env
# Azure Blob Storage
AZURE_STORAGE_CONNECTION_STRING=your_connection_string
AZURE_CONTAINER_NAME=datasets

# Azure Cognitive Services (for Gold layer)
AZURE_LANGUAGE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_LANGUAGE_KEY=your_api_key

# Supabase Database
SUPABASE_DB_URL=postgresql://user:password@host:port/database?sslmode=require
DEST_TABLE=public.hotels_gold
```

### 2. Data Files

Ensure Excel files are in the `Datasets/` folder:
- `Hotel_Torre_Mar.xlsx`
- `Hotel_Steven_Buenaventura.xlsx`
- `Hotel_Maguipi.xlsx`
- `Hotel_Cordillera_Buenaventura.xlsx`
- `Cosmos_Pacifico_Hotel.xlsx`

### 3. Dependencies

Install all required packages:
```powershell
python -m pip install -r requirements.txt
```

## Step-by-Step Execution

### Step 1: Bronze → Silver Layer

**Purpose**: Transform raw Excel files into cleaned Silver layer data

**Command**:
```powershell
python -m src.silver_build
```

**What it does**:
- Reads Excel files from `Datasets/` folder
- Applies data cleaning and transformation
- Filters for Google reviews (2020-2025)
- Deduplicates by reviewId
- Uploads to Azure Blob Storage as `silver/{hotel_id}_silver.parquet`

**Expected Output**:
```
→ Reading Hotel_Torre_Mar.xlsx (sheet=Data)
🧩 Filtered by Google origin: 1500 → 1450
🕓 Filtered by date: 1450 → 1200 (range 2020-01-01 to 2025-08-21)
✅ Final Silver for 'torre_mar': 1200 valid Google reviews within date range.
📊 Silver shape: (1200, 38)
☁️ Uploaded Silver: silver/torre_mar_silver.parquet (1200 rows)
...
```

**Verify**:
- Check Azure Blob Storage for `silver/*.parquet` files
- Or use: `python -m src.list_blob` to list files

---

### Step 2: Silver → Gold Layer

**Purpose**: Enrich Silver data with sentiment analysis and aspect extraction

**Command** (Cloud mode - uses Azure API):
```powershell
python -m src.gold_build --hotel all --mode cloud --language es
```

**Command** (Local mode - mock data for testing):
```powershell
python -m src.gold_build --hotel all --mode local
```

**What it does**:
- Loads Silver parquet files from Azure
- Enriches with Azure Cognitive Services:
  - Sentiment analysis (positive/neutral/negative)
  - Aspect extraction (opinion mining)
  - Key phrases extraction
  - Entity recognition
  - PII detection
- Normalizes aspects to thematic categories
- Filters out "Mixed" sentiment reviews
- Uploads to Azure Blob Storage as `gold/{hotel_id}_GOLD.parquet`

**Expected Output**:
```
🟨 Processing hotel: torre_mar
SILVER → (1200, 38) rows
🧠 Processing 1200 valid reviews with Azure Language Service...
✅ Sentiment completed: 1200 reviews processed
✅ Key Phrases completed: 1200 reviews
✅ Entities completed: 1200 reviews
✅ Linked Entities completed: 1200 reviews
✅ PII completed: 1200 reviews
🧹 Filtered reviews with 'Mixed' sentiment: 5 of 1200
🔄 Normalizing aspects to canonical names...
✅ Aspect normalization complete: 12 unique themes, 0 unclassified
✅ Enrichment complete: 1200 reviews analyzed with Azure.
GOLD ready → (1195, 45) rows
☁️ Uploaded Gold: gold/torre_mar_GOLD.parquet (1195 rows)
...
```

**Options**:
- `--hotel all`: Process all hotels (or specify: `--hotel torre_mar`)
- `--mode cloud`: Use Azure API (requires credentials)
- `--mode local`: Use mock data (for testing)
- `--language es`: Language code (or `none` for auto-detect)

**Verify**:
- Check Azure Blob Storage for `gold/*_GOLD.parquet` files
- Gold files should have additional columns: `aspect_raw`, `aspect_theme`, sentiment scores, etc.

---

### Step 3: Gold → Database (Supabase)

**Purpose**: Load Gold data into Supabase database for analytics

**Command**:
```powershell
python -m src.nashor_to_supabase
```

**Command** (with options):
```powershell
# Process all Gold files
python -m src.nashor_to_supabase

# Process single file
python -m src.nashor_to_supabase --only torre_mar_GOLD.parquet

# Truncate table before loading (fresh start)
python -m src.nashor_to_supabase --truncate

# Custom batch size
python -m src.nashor_to_supabase --batch 500
```

**What it does**:
- Loads all Gold parquet files from Azure
- Unions and deduplicates by reviewId
- Normalizes data types for database
- Maps columns to database schema
- Performs ACID-compliant batch upserts
- Commits transactions

**Expected Output**:
```
🔌 Pooler: host:6543/database
✅ Pooler connection OK.
📦 Files detected: 5
🧭 Union and deduplication by reviewId ...
   - Unique rows: 3700
🔎 PK detected: reviewid
🧩 Columns to insert (45): hotel_id, reviewid, placeid, ...
   ✓ Batch 1: 200 rows (total 200/3700)
   ✓ Batch 2: 200 rows (total 400/3700)
   ...
✅ All transactions committed successfully.
🎉 Unified load completed.
```

**Options**:
- `--only <filename>`: Process single parquet file
- `--truncate`: Empty table before loading
- `--batch <size>`: Batch size for upserts (default: 200)

**Verify**:
- Check Supabase database table `hotels_gold`
- Query: `SELECT COUNT(*) FROM hotels_gold;`
- Verify columns: `aspect_raw`, `aspect_theme` exist

---

## Complete Pipeline Example

Run all three steps in sequence:

```powershell
# Step 1: Bronze → Silver
python -m src.silver_build

# Step 2: Silver → Gold (with Azure API)
python -m src.gold_build --hotel all --mode cloud --language es

# Step 3: Gold → Database
python -m src.nashor_to_supabase
```

## Troubleshooting

### Issue: "No module named 'sqlalchemy'"
**Solution**: Install dependencies
```powershell
python -m pip install sqlalchemy psycopg psycopg-binary azure-storage-blob azure-ai-textanalytics
```

### Issue: "Missing AZURE_LANGUAGE_ENDPOINT"
**Solution**: Add to `.env` file:
```env
AZURE_LANGUAGE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_LANGUAGE_KEY=your_key
```

### Issue: "Could not read silver/{hotel_id}.parquet"
**Solution**: Check that Silver files exist in Azure Blob Storage. Run Step 1 first.

### Issue: "No silver/*.parquet files found"
**Solution**: 
- Verify Azure connection string in `.env`
- Check that Silver layer was created successfully
- List files: `python -m src.list_blob`

### Issue: Database connection errors
**Solution**:
- Verify `SUPABASE_DB_URL` in `.env`
- Ensure URL uses pooler port (6543) if using pooler
- Check SSL mode is set: `?sslmode=require`

## Testing Individual Steps

### Test Silver Layer (Local)
```powershell
python -m src.silver_build
```

### Test Gold Layer (Local - Mock Data)
```powershell
python -m src.gold_build --hotel torre_mar --mode local
```

### Test Database Connection
```powershell
python -c "from src.infrastructure.database import ping; print(ping())"
```

## Data Flow Summary

```
Excel Files (Datasets/)
    ↓
[Step 1: silver_build.py]
    ↓
Silver Parquet (Azure: silver/*.parquet)
    ↓
[Step 2: gold_build.py]
    ↓
Gold Parquet (Azure: gold/*_GOLD.parquet)
    ↓
[Step 3: nashor_to_supabase.py]
    ↓
Supabase Database (hotels_gold table)
```

## Output Columns

After complete pipeline, the database table contains:

**Original Columns**:
- `hotel_id`, `reviewId`, `text`, `stars`, `publishedAtDate`, etc.

**Enriched Columns** (from Gold layer):
- `sentiment_label`, `positive_score`, `neutral_score`, `negative_score`
- `sentiment_score`, `sentences_count`
- `aspects` (original Azure format)
- `aspect_raw` (extracted raw aspects)
- `aspect_theme` (thematic categories - for analytics)
- `key_phrases`, `entities`, `pii_entities`, `linked_entities`
- `scored_at` (timestamp of enrichment)

## Next Steps

After loading data:
1. Verify data in Supabase: `SELECT * FROM hotels_gold LIMIT 10;`
2. Check aspect themes: `SELECT DISTINCT aspect_theme FROM hotels_gold;`
3. Create analytical views for Power BI
4. Build dashboards using `aspect_theme` column for consistent grouping

