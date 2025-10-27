# 🏨 Buenaventura Hotels – Sentiment Analysis Pipeline

![Pipeline Overview](<img src="https://raw.githubusercontent.com/Pipe199x/Buenaventura-Hotels/main/docs/images/pipeline_overview.png" alt="Pipeline Overview" width="500">)

> **End-to-end sentiment analysis system for hotel reviews in Buenaventura (Colombia)**
> Built with **Azure**, **Supabase**, and **Power BI** — transforming raw Google Maps data into analytical insights for tourism development.

---

## 🌊 Project Overview

The **Buenaventura Hotels** project aims to analyze and visualize the sentiment of hotel reviews from the Buenaventura region.
It uses a full **ETL + NLP pipeline** hosted on Azure, enriched by **AI language services**, and visualized in **Power BI** dashboards.

This project forms part of the undergraduate **Data Analytics Engineering Capstone** at the **University of Manizales**.

---

## 🧱 Architecture

```
📂 data/
 ┗ raw/        → Original hotel Excel files (per hotel)
 ┗ silver/     → Cleaned & filtered reviews (Google-only)
 ┗ gold/       → Enriched data (Azure Text Analytics)
 ┗ nashor/     → Final dataset uploaded to Supabase
```

**Main transformation flow:**

```
RAW  →  SILVER  →  GOLD  →  NASHOR  →  POWER BI
Excel   Clean    Enrich   Upload     Visualize
```

---

## ⚙️ Stack Overview

| Layer            | Technology                | Description                                   |
| ---------------- | ------------------------- | --------------------------------------------- |
| ☁️ Cloud         | **Azure Storage**         | Data lake structure (`raw/silver/gold`)       |
| 🧠 AI / NLP      | **Azure Text Analytics**  | Sentiment, aspects, key phrases, entities     |
| 🐍 Backend       | **Python (ETL)**          | Data cleaning, enrichment, and upload scripts |
| 🧰 Database      | **Supabase (PostgreSQL)** | Central repository for Power BI               |
| 📊 Visualization | **Power BI**              | Dashboards and trend reports                  |
| 🔄 Automation    | **GitHub Actions**        | Scheduled ETL and updates                     |

---

## 🧉 Key Components

### 1️⃣ `silver_build.py`

Cleans and filters raw hotel data:

* Keeps only **Google reviews**
* Filters dates (`2020-01-01` → `2025-08-21`)
* Standardizes columns and review IDs

### 2️⃣ `gold_build.py`

Enriches reviews using **Azure Cognitive Services**:

* Adds sentiment (`positive`, `neutral`, `negative`)
* Extracts **aspects**, **entities**, and **key phrases**
* Produces clean, analytics-ready Parquet files

### 3️⃣ `nashor_to_supabase.py`

Uploads all `gold` files to **Supabase**:

* Merges datasets automatically
* Cleans invalid JSON (NaN, inf, timestamps)
* Inserts in **batches of 500 rows**
* Updates the `hotels_gold` table

### 4️⃣ `create_views_supabase.py`

Creates analytical views directly in Supabase:

* `vw_sentiment_summary` → Avg sentiment & stars per hotel
* `vw_missing_reviews` → Reviews with missing text/translation
* `vw_review_distribution` → Reviews by year and star rating

---

## 🧮 Database Schema (Supabase)

```sql
CREATE TABLE hotels_gold (
  hotel_id TEXT,
  reviewId TEXT PRIMARY KEY,
  placeId TEXT,
  title TEXT,
  text TEXT,
  textTranslated TEXT,
  originalLanguage TEXT,
  reviewOrigin TEXT,
  publishedAtDate TIMESTAMPTZ,
  year_month TEXT,
  stars FLOAT,
  totalScore FLOAT,
  reviewsCount FLOAT,
  hotelStars FLOAT,
  price FLOAT,
  isLocalGuide BOOLEAN,
  reviewerNumberOfReviews FLOAT,
  likesCount FLOAT,
  responseFromOwnerText TEXT,
  responseFromOwnerDate TIMESTAMPTZ,
  response_delay_days FLOAT,
  review_length INT,
  scrapedAt TIMESTAMPTZ,
  categoryName TEXT,
  reviewUrl TEXT,
  url TEXT,
  sentiment_label TEXT,
  positive_score FLOAT,
  neutral_score FLOAT,
  negative_score FLOAT,
  sentiment_score FLOAT,
  scored_at TIMESTAMPTZ,
  hotel_name TEXT
);
```

---

## 📊 Power BI Dashboard

![Power BI Screenshot](https://raw.githubusercontent.com/Pipe199x/Buenaventura-Hotels/main/docs/images/powerbi_dashboard.png)

### Key visuals:

* 📊 Average sentiment per hotel
* ⭐ Review star distribution
* 🕓 Sentiment trend over time
* 🌍 Map of hotel review density

---

## 🧠 Example Analytical View

```sql
CREATE OR REPLACE VIEW vw_sentiment_summary AS
SELECT
    hotel_id,
    hotel_name,
    COUNT(reviewId) AS total_reviews,
    ROUND(AVG(stars), 2) AS avg_stars,
    ROUND(AVG(sentiment_score), 3) AS avg_sentiment
FROM hotels_gold
GROUP BY hotel_id, hotel_name;
```

---

## 🚀 Quick Start

```bash
# 1️⃣ Clone the repository
git clone https://github.com/Pipe199x/Buenaventura-Hotels.git
cd Buenaventura-Hotels

# 2️⃣ Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run ETL pipeline
python -m src.silver_build
python -m src.gold_build
python -m src.nashor_to_supabase

# 5️⃣ Create analytical views
python -m src.create_views_supabase
```

---

## 🔐 Environment Variables

Make sure you have a `.env` file in the root directory:

```bash
AZURE_STORAGE_CONNECTION_STRING=<your_azure_connection>
AZURE_CONTAINER_NAME=datasets

SUPABASE_URL=https://<project>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<your_service_role_key>
```

---

## 🦯 Future Roadmap

* [ ] Add automated daily ETL via **GitHub Actions**
* [ ] Incorporate **topic modeling** with Azure or spaCy
* [ ] Create hotel-level anomaly detection (sudden rating drops)
* [ ] Expand to **restaurants and tourism reviews**
* [ ] Publish public Power BI report

---

## 🖼️ Visual Architecture

![Data Flow Diagram]<img src="https://raw.githubusercontent.com/Pipe199x/Buenaventura-Hotels/main/docs/images/dataflow_diagram.png" alt="Data Flow" width="300">

---

## 💡 Key Insight

> "Sentiment data is more than opinion — it’s a real-time indicator of service quality and regional development potential."

---

## 💬 Contact

📧 [duqueandres800@gmail.com](mailto:duqueandres800@gmail.com)
🐙 [GitHub: Pipe199x](https://github.com/Pipe199x)

---

⭐ **If you find this project helpful, give it a star!**
It helps others discover the work and supports open-data tourism research 🌍
