# 🏨 Sentiment Analysis of Hotel Reviews in Buenaventura (2020–2025)

This repository contains the full data analytics pipeline, models, and visual analytics developed for the academic research project:

> **“Sentiment Analysis of Hotel Reviews in Buenaventura using Text Mining and Data Analytics”**

The project applies **text mining**, **automated sentiment analysis**, and **data visualization** techniques to understand how tourists perceive the hotel offer in Buenaventura, Colombia, based on Google reviews published between **2020 and 2025**.

---

## 📌 Project Objectives

### General Objective
Analyze tourist perception of the main hotels in Buenaventura through sentiment analysis of online reviews, identifying emotional patterns, recurring service aspects, and trends over time to support data-driven decision making.

### Specific Objectives
1. Collect and preprocess hotel reviews from Google Maps using automated scraping techniques.
2. Apply NLP-based sentiment analysis to classify polarity and extract service aspects.
3. Visualize statistical patterns and sentiment trends using KPIs and advanced dashboards.
4. Identify persistent negative aspects over time to support improvement strategies for hotels.

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
   Apply sentiment analysis and aspect extraction using AI services.

5. **Evaluation**  
   Validate sentiment coherence through manual sampling (≈3% of reviews).

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

Total analyzed reviews: **3,700+**

---

## ⚙️ Technical Architecture

### 🔹 Data Ingestion
- **Apify – Google Maps Reviews Scraper**
- Automated extraction of structured and unstructured review data

### 🔹 Storage & Processing
- **Azure Blob Storage**
  - Bronze / Silver / Gold layers
- **Supabase (PostgreSQL)**
  - Final analytical storage
  - SQL views optimized for BI consumption

### 🔹 Semantic Enrichment
- **Azure Cognitive Services – Text Analytics**
  - Sentiment polarity (positive / neutral / negative)
  - Confidence scores
  - Aspect extraction
  - Key phrases and entities

### 🔹 Analytics & Visualization
- **Power BI**
  - Executive KPIs
  - Sentiment distribution
  - Aspect-level analysis
  - Temporal trend analysis
  - Interactive tooltips with real review text

---

## 🧩 Database Model (Core Table)

### `hotels_gold`
Central analytical table containing:
- Review metadata (hotel, date, rating, language)
- Cleaned and translated text
- Sentiment labels and scores
- Extracted aspects and entities
- Owner response data (response text, delay)

---

## 📊 Key Analytical SQL Views

| View Name | Description |
|---------|-------------|
| `vw_sentiment_overview_summary` | Global sentiment KPIs per hotel |
| `vw_sentiment_polarity_detailed` | Monthly sentiment breakdown |
| `vw_sentiment_aspect_analysis` | Aspect-level sentiment analysis |
| `vw_aspects_by_year` | Negative aspects frequency per year |
| `vw_top_negative_aspects` | Most recurrent negative aspects |

These views are optimized for **direct Power BI consumption**.

---

## 📈 Power BI Dashboards

Main dashboards include:

- **Executive Overview**
  - Average rating
  - Sentiment score
  - Review volume
  - Correlation metrics

- **Aspect Criticism Matrix (2020–2025)**
  - Identifies recurring negative aspects per year
  - Conditional formatting with alert icons
  - KPIs such as *“Most criticized aspect of the year”*

- **Interactive Tooltips**
  - Display real negative review excerpts on hover
  - Enables qualitative validation of quantitative metrics

---

## 🧪 Validation Strategy

- Manual review of ~3% of reviews
- Comparison between predicted sentiment and human interpretation
- Achieved **≈90% interpretative coherence**

---

## 🧠 Key Insights

- Hotels with high coherence between rating and sentiment project a stronger and more reliable tourist image.
- Persistent negative aspects (e.g. service, prices, attention) can be tracked longitudinally.
- High review volume does not guarantee positive perception.
- Textual analysis complements numeric ratings by revealing emotional drivers.

---

## 🚀 Technologies Used

- **Python** (pandas, NLP preprocessing)
- **SQL (PostgreSQL / Supabase)**
- **Azure Cognitive Services**
- **Azure Blob Storage**
- **Apify**
- **Power BI**
- **CRISP-DM methodology**

---

## 📄 Academic Context

This repository is part of an undergraduate research project in **Data Analytics Engineering**, developed as a graduation requirement.

The project is designed to be:
- Reproducible
- Auditable
- Scalable
- Suitable for academic and professional evaluation

---

## 👤 Author

**Andrés Felipe Duque Caicedo**  
Data Analytics Engineer  
GitHub: [Pipe199x](https://github.com/Pipe199x)

---

## 📜 License

This project is released for academic and educational purposes.  
For commercial use, please contact the author.

