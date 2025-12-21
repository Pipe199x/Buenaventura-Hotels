# 🏨 Sentiment Analysis of Hotel Reviews in Buenaventura (2020–2025)

This repository contains a complete data analytics pipeline for sentiment analysis and thematic categorization of hotel reviews in Buenaventura, Colombia.

> **"Sentiment Analysis of Hotel Reviews in Buenaventura using Text Mining and Data Analytics"**

The project applies **text mining**, **automated sentiment analysis**, and **thematic aspect normalization** to understand tourist perception of hotels based on Google reviews published between **2020 and 2025**.

---

## 📌 Project Objectives

### General Objective
Analyze tourist perception of the main hotels in Buenaventura through sentiment analysis of online reviews, identifying emotional patterns, recurring service aspects, and trends over time to support data-driven decision making.

### Specific Objectives
1. Collect and preprocess hotel reviews from Google Maps using automated scraping techniques.
2. Apply NLP-based sentiment analysis to classify polarity and extract service aspects.
3. Normalize extracted aspects into consistent thematic categories for analytical reporting.
4. Visualize statistical patterns and sentiment trends using KPIs and advanced dashboards.
5. Identify persistent negative aspects over time to support improvement strategies for hotels.

---

## 🧠 Methodological Approach

This project follows the **CRISP-DM methodology**, structured as follows:

1. **Business Understanding**  
   Define research questions related to tourist perception and hotel image.

2. **Data Understanding**  
   Analyze raw Google reviews: languages, text availability, ratings, volume, and temporal distribution.

3. **Data Preparation**  
   Clean, normalize, translate (if needed), and enrich text data for NLP processing.

4. **Modeling**  
   Apply sentiment analysis and aspect extraction using Azure Cognitive Services, then normalize aspects into thematic categories.

5. **Evaluation**  
   Validate sentiment coherence through manual sampling (≈3% of reviews) and thematic categorization accuracy.

6. **Deployment**  
   Deliver insights via SQL analytical views and Power BI dashboards.

---

## 🗂️ Data Sources

- **Google Maps Reviews**
- Time range: **2020 – 2025**
- Hotels analyzed:
  - Hotel Cordillera
  - Hotel Cosmos Pacífico
  - Hotel Magüipí
  - Hotel Torre Mar
  - Hotel Steven Buenaventura

**Total analyzed reviews: 3,700+**

---

## ⚙️ Technical Architecture

### 🔹 Data Pipeline (Three-Layer Architecture)

The pipeline follows a **Bronze → Silver → Gold** architecture:

#### **Bronze Layer** (Raw Data)
- Raw Excel files from Google Maps scraping
- Stored in Azure Blob Storage: `bronze/`
- No transformations applied

#### **Silver Layer** (Cleaned Data)
- Data cleaning and normalization
- Filtering for Google reviews (2020-2025)
- Deduplication by `reviewId`
- Stored in Azure Blob Storage: `silver/{hotel_id}_silver.parquet`

#### **Gold Layer** (Enriched Data)
- Sentiment analysis via Azure Cognitive Services
- Aspect extraction (service aspects mentioned in reviews)
- Thematic aspect normalization (15 thematic categories)
- Key phrases, entities, and PII detection
- Stored in Azure Blob Storage: `gold/{hotel_id}_GOLD.parquet`

### 🔹 Code Architecture (Layered Design)

The codebase follows a **clean architecture** pattern:

```
src/
├── infrastructure/     # External services (Azure, Database)
│   ├── blob_storage.py
│   ├── database.py
│   └── config.py
├── domain/            # Business logic
│   ├── transformations.py      # Bronze→Silver, Silver→Gold
│   ├── aspect_mappings.py      # Thematic category mappings
│   └── aspect_normalization.py # Aspect normalization logic
├── application/       # Application services (future)
└── interfaces/        # API/CLI interfaces (future)
```

### 🔹 Storage & Processing

- **Azure Blob Storage**
  - Bronze / Silver / Gold layers
  - Parquet format for efficient storage

- **Supabase (PostgreSQL)**
  - Final analytical storage
  - SQL views optimized for BI consumption
  - ACID-compliant transactions

### 🔹 Semantic Enrichment

- **Azure Cognitive Services – Text Analytics**
  - Sentiment polarity (positive / neutral / negative)
  - Confidence scores
  - Aspect extraction (raw aspects from text)
  - Key phrases and entities
  - PII detection

### 🔹 Aspect Normalization System

**Two-Level Model:**
- `aspect_raw`: Original extracted text from Azure
- `aspect_theme`: Normalized thematic category for analytics

**15 Thematic Categories:**
1. `service_quality` - General service quality
2. `staff_attention` - Staff treatment and attention
3. `rooms_accommodation` - Rooms and accommodation
4. `bathrooms_cleanliness` - Bathrooms and cleanliness
5. `food_dining` - Food and restaurant
6. `maintenance_facilities` - Maintenance and facilities
7. `infrastructure_amenities` - Infrastructure and amenities
8. `cleanliness_general` - General cleanliness
9. `pricing_value` - Pricing and value
10. `connectivity_technology` - Connectivity and technology
11. `guest_experience` - Overall guest experience
12. `location_surroundings` - Location and surroundings
13. `safety_security` - Safety and security
14. `noise_quietness` - Noise and quietness
15. `comfort_furnishings` - Comfort and furnishings

**Features:**
- Comprehensive Spanish variant mapping (singular/plural, synonyms, accents)
- No generic "catch-all" categories - all aspects map to meaningful themes
- Consistent categorization across all hotels
- Stable for year-by-year comparisons

### 🔹 Analytics & Visualization

- **Power BI**
  - Executive KPIs
  - Sentiment distribution
  - Thematic aspect analysis
  - Temporal trend analysis
  - Interactive tooltips with real review text

---

## 🧩 Database Model

### Core Table: `hotels_gold` (or `parsed`)

Central analytical table containing:
- Review metadata (hotel, date, rating, language)
- Cleaned and translated text (`text_used`)
- Sentiment labels and scores (`sentiment_label`, `sentiment_score`)
- Extracted aspects (`aspects`, `aspect_raw`, `aspect_theme`)
- Key phrases, entities, and PII
- Owner response data (response text, delay)
- Year/month columns for temporal analysis

---

## 📊 Key Analytical SQL Views

| View Name | Description |
|---------|-------------|
| `vw_sentiment_overview_summary` | Global sentiment KPIs per hotel |
| `vw_sentiment_polarity_detailed` | Monthly sentiment breakdown |
| `vw_sentiment_aspect_analysis` | Aspect-level sentiment analysis |
| `vw_aspects_by_year` | Thematic aspects frequency per year |
| `vw_top_negative_aspects` | Most recurrent negative thematic aspects |

These views are optimized for **direct Power BI consumption**.

---

## 🚀 Quick Start

### Prerequisites

1. **Python 3.9+**
2. **Azure Account** (for Blob Storage and Cognitive Services)
3. **Supabase Account** (PostgreSQL database)
4. **Environment Variables** (`.env` file)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd "Proyecto De Grado"
   ```

2. **Install dependencies:**
   ```bash
   python -m pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   Create a `.env` file in the project root:
   ```env
   # Azure Blob Storage
   AZURE_STORAGE_CONNECTION_STRING=your_connection_string
   AZURE_CONTAINER_NAME=datasets
   
   # Azure Cognitive Services
   AZURE_LANGUAGE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
   AZURE_LANGUAGE_KEY=your_api_key
   
   # Supabase Database
   SUPABASE_DB_URL=postgresql://user:password@host:port/database?sslmode=require
   DEST_TABLE=public.hotels_gold
   
   # Optional: Database Pooler
   USE_POOLER=false
   POOLER_HOST=your-pooler-host
   POOLER_PORT=6543
   ```

4. **Prepare data files:**
   Place Excel files in the `Datasets/` folder:
   - `Hotel_Torre_Mar.xlsx`
   - `Hotel_Steven_Buenaventura.xlsx`
   - `Hotel_Maguipi.xlsx`
   - `Hotel_Cordillera_Buenaventura.xlsx`
   - `Cosmos_Pacifico_Hotel.xlsx`

### Running the Pipeline

#### Step 1: Bronze → Silver
```bash
python -m src.silver_build
```
Transforms raw Excel files into cleaned Silver layer data.

#### Step 2: Silver → Gold
```bash
python -m src.gold_build --hotel cordillera --language es
```
Enriches Silver data with Azure Cognitive Services (sentiment, aspects).

#### Step 3: Gold → Database
```bash
python -m src.nashor_to_supabase --hotel cordillera
```
Loads enriched Gold data into Supabase PostgreSQL.

**For detailed execution guide, see:** [`docs/PIPELINE_EXECUTION_GUIDE.md`](docs/PIPELINE_EXECUTION_GUIDE.md)

---

## 📈 Power BI Dashboards

Main dashboards include:

- **Executive Overview**
  - Average rating
  - Sentiment score
  - Review volume
  - Correlation metrics

- **Thematic Aspect Analysis (2020–2025)**
  - Identifies recurring thematic aspects per year
  - Separates positive and negative sentiment
  - Conditional formatting with alert icons
  - KPIs such as *"Most criticized thematic aspect of the year"*

- **Interactive Tooltips**
  - Display real review excerpts on hover
  - Enables qualitative validation of quantitative metrics

---

## 🧪 Validation Strategy

- Manual review of ~3% of reviews
- Comparison between predicted sentiment and human interpretation
- Achieved **≈90% interpretative coherence**
- Thematic categorization validated against manual analysis tables

---

## 🧠 Key Insights

- Hotels with high coherence between rating and sentiment project a stronger and more reliable tourist image.
- Persistent negative thematic aspects (e.g. service, prices, attention) can be tracked longitudinally.
- High review volume does not guarantee positive perception.
- Textual analysis complements numeric ratings by revealing emotional drivers.
- Thematic normalization enables consistent year-by-year comparisons across all hotels.

---

## 🛠️ Technologies Used

### Core Technologies
- **Python 3.9+** (pandas, numpy)
- **SQL (PostgreSQL / Supabase)**
- **Azure Cognitive Services** (Text Analytics API)
- **Azure Blob Storage**
- **SQLAlchemy** (database ORM)
- **Psycopg** (PostgreSQL driver)

### Data Formats
- **Excel** (`.xlsx`) - Raw data input
- **Parquet** (`.parquet`) - Processed data storage
- **PostgreSQL** - Final analytical database

### Development Tools
- **Jupyter Notebooks** - Data exploration
- **Power BI** - Visualization and dashboards
- **Git** - Version control

### Methodology
- **CRISP-DM** - Data mining methodology
- **Clean Architecture** - Code organization
- **ACID Principles** - Database transactions

---

## 📚 Documentation

- [`docs/PIPELINE_EXECUTION_GUIDE.md`](docs/PIPELINE_EXECUTION_GUIDE.md) - Complete pipeline execution guide
- [`docs/THEMATIC_ASPECT_NORMALIZATION.md`](docs/THEMATIC_ASPECT_NORMALIZATION.md) - Aspect normalization system documentation
- [`docs/ASPECT_MAPPINGS_UPDATE.md`](docs/ASPECT_MAPPINGS_UPDATE.md) - Aspect mappings update documentation
- [`docs/THEMATIC_TABLE_CREATION.md`](docs/THEMATIC_TABLE_CREATION.md) - Guide for creating thematic tables

---

## 📄 Academic Context

This repository is part of an undergraduate research project in **Data Analytics Engineering**, developed as a graduation requirement.

The project is designed to be:
- **Reproducible** - Clear documentation and version control
- **Auditable** - Transparent data processing steps
- **Scalable** - Layered architecture supports growth
- **Maintainable** - Clean code structure and separation of concerns
- **Suitable for academic and professional evaluation**

---

## 👤 Author

**Andrés Felipe Duque Caicedo**  
Data Analytics Engineer  
GitHub: [Pipe199x](https://github.com/Pipe199x)

---

## 📜 License

This project is released for academic and educational purposes.  
For commercial use, please contact the author.

---

## 🙏 Acknowledgments

- Azure Cognitive Services for sentiment analysis and aspect extraction
- Supabase for PostgreSQL hosting
- Power BI for visualization capabilities
