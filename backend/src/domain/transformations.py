"""Data transformation functions for Bronze, Silver, and Gold layers."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
import pandas as pd
import numpy as np


def make_silver(df: pd.DataFrame, hotel_id: str) -> pd.DataFrame:
    """
    Transform raw data to Silver layer format.
    
    Applies data cleaning, type conversion, filtering, and deduplication.
    
    Args:
        df: Raw DataFrame from Bronze layer
        hotel_id: Identifier for the hotel
        
    Returns:
        Cleaned DataFrame in Silver format
    """
    df = df.copy()

    base = [
        "reviewId","placeId","title","text","textTranslated","originalLanguage","reviewOrigin",
        "publishedAtDate","stars",
        "totalScore","reviewsCount","hotelStars","price",
        "isLocalGuide","reviewerNumberOfReviews","likesCount",
        "responseFromOwnerText","responseFromOwnerDate",
        "scrapedAt","categoryName","reviewUrl","url","source"
    ]
    ctx = [c for c in df.columns if c.startswith("reviewContext/")]
    det = [c for c in df.columns if c.startswith("reviewDetailedRating/")]
    keep = [c for c in base if c in df.columns] + sorted(ctx) + sorted(det)
    df = df.loc[:, keep].copy()

    # Convert dates to UTC
    for c in ["publishedAtDate","responseFromOwnerDate","scrapedAt"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)

    # Convert to numeric
    for c in ["stars","reviewsCount","hotelStars","reviewerNumberOfReviews","likesCount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in det:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Consistent boolean
    if "isLocalGuide" in df.columns:
        df["isLocalGuide"] = df["isLocalGuide"].map(
            {True: True, False: False, "true": True, "false": False, "True": True, "False": False}
        )

    # Deduplicate by reviewId (keep the most recent)
    if "reviewId" in df.columns:
        sort_keys = [c for c in ["scrapedAt","publishedAtDate"] if c in df.columns]
        if sort_keys:
            df = df.sort_values(sort_keys).drop_duplicates(subset=["reviewId"], keep="last")

    # Filter 1 - only Google reviews
    before_origin = len(df)
    df = df[df["reviewOrigin"].fillna("").str.lower() == "google"]
    after_origin = len(df)
    print(f"🧩 Filtered by Google origin: {before_origin} → {after_origin}")

    # Filter 2 - date range 2020–current
    if "publishedAtDate" in df.columns:
        start = pd.Timestamp("2020-01-01", tz="UTC")
        end = pd.Timestamp("2025-08-21", tz="UTC")
        before_date = len(df)
        df = df.loc[df["publishedAtDate"].between(start, end, inclusive="both")].copy()
        after_date = len(df)
        print(f"🕓 Filtered by date: {before_date} → {after_date} (range {start.date()} to {end.date()})")

    # Useful derived fields
    if "publishedAtDate" in df.columns:
        df["year_month"] = df["publishedAtDate"].dt.to_period("M").astype(str)
    else:
        df["year_month"] = pd.NA

    if "text" in df.columns:
        df["review_length"] = df["text"].fillna("").astype(str).str.len()

    if {"publishedAtDate","responseFromOwnerDate"}.issubset(df.columns):
        delay = (df["responseFromOwnerDate"] - df["publishedAtDate"]).dt.days
        df["response_delay_days"] = delay.where(delay >= 0)

    df["hotel_id"] = hotel_id

    preferred = [
        "hotel_id","reviewId","placeId","title","text","textTranslated","originalLanguage","reviewOrigin",
        "publishedAtDate","year_month","stars","totalScore","reviewsCount","hotelStars","price",
        "isLocalGuide","reviewerNumberOfReviews","likesCount",
        "responseFromOwnerText","responseFromOwnerDate","response_delay_days",
        "review_length","scrapedAt","categoryName"
    ]
    ordered = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]

    print(f"✅ Final Silver for '{hotel_id}': {len(df)} valid Google reviews within date range.")
    return df[ordered]


def enrich_with_azure(df_silver: pd.DataFrame, *, language: Optional[str] = "es", mode: str = "cloud") -> pd.DataFrame:
    """
    Enrich Silver data with Azure Cognitive Services sentiment analysis.
    
    Args:
        df_silver: Silver layer DataFrame
        language: Language code for analysis (default: "es")
        mode: "cloud" for Azure API or "local" for mock data
        
    Returns:
        Enriched DataFrame with sentiment and NLP features
    """
    # Import here to avoid circular dependencies
    from azure.core.credentials import AzureKeyCredential
    from azure.ai.textanalytics import TextAnalyticsClient
    import time
    import os
    from dotenv import load_dotenv
    
    df = df_silver.copy()

    # Prepare valid text
    df["text_used"] = df.get("text", "").fillna("").astype(str).str.strip()
    mask_empty = df["text_used"] == ""
    if "textTranslated" in df.columns:
        df.loc[mask_empty, "text_used"] = df.loc[mask_empty, "textTranslated"].fillna("").astype(str).str.strip()

    # Initialize columns
    add_cols = [
        "sentiment_label", "positive_score", "neutral_score", "negative_score",
        "sentiment_score", "sentences_count", "aspects",
        "key_phrases", "entities", "pii_entities", "linked_entities",
        "scored_at"
    ]
    for c in add_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Local mode - mock data
    if mode == "local":
        np.random.seed(42)
        df["sentiment_label"] = np.random.choice(["positive", "neutral", "negative"], len(df))
        df["positive_score"] = np.random.uniform(0.5, 1.0, len(df))
        df["neutral_score"] = np.random.uniform(0.0, 0.5, len(df))
        df["negative_score"] = 1 - df["positive_score"]
        df["sentiment_score"] = (df["positive_score"] - df["negative_score"] + 1) / 2
        df["sentences_count"] = np.random.randint(1, 3, len(df))
        df["aspects"] = "mock_aspects"
        df["key_phrases"] = "mock_keyphrases"
        df["entities"] = "mock_entities"
        df["linked_entities"] = "mock_linked"
        df["pii_entities"] = "mock_pii"
        df["scored_at"] = datetime.now(timezone.utc)
        return df

    # Cloud mode - Azure API
    load_dotenv()
    
    LANG_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT")
    LANG_KEY = os.getenv("AZURE_LANGUAGE_KEY")
    
    if not LANG_ENDPOINT or not LANG_KEY:
        raise SystemExit("❌ Missing AZURE_LANGUAGE_ENDPOINT or AZURE_LANGUAGE_KEY in .env")
    
    ta = TextAnalyticsClient(endpoint=LANG_ENDPOINT, credential=AzureKeyCredential(LANG_KEY))
    lang_param = None if language is None else language
    mask_valid = df["text_used"].fillna("").astype(str).str.strip().str.len() > 0
    valid_idx = df.index[mask_valid].tolist()
    texts = df.loc[mask_valid, "text_used"].tolist()

    print(f"🧠 Processing {len(texts)} valid reviews with Azure Language Service...")

    BATCH_SIZE = 5  # Maximum allowed by Azure

    for col in ["sentiment_label", "aspects", "key_phrases"]:
        if col in df.columns:
            df[col] = df[col].astype(object)

    # Sentiment analysis
    sent_rows = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        for attempt in range(4):
            try:
                resp = ta.analyze_sentiment(batch, language=lang_param, show_opinion_mining=True)
                sent_rows.extend(resp)
                break
            except Exception as e:
                print(f"⚠️ Error in Sentiment (attempt {attempt+1}/4): {e}")
                time.sleep(1.2 * (attempt + 1))

    valid_results = [r for r in sent_rows if not getattr(r, "is_error", False)]
    lab, pos, neu, neg, sc, nsents, aspects = [], [], [], [], [], [], []
    for r in valid_results:
        lab.append(str(r.sentiment))
        pos.append(float(r.confidence_scores.positive))
        neu.append(float(r.confidence_scores.neutral))
        neg.append(float(r.confidence_scores.negative))
        sc.append((pos[-1] - neg[-1] + 1) / 2)
        nsents.append(len(r.sentences))
        pairs = []
        for s in r.sentences:
            for mo in getattr(s, "mined_opinions", []):
                target = mo.target.text
                snt = mo.target.sentiment
                ops = ", ".join([a.text for a in mo.assessments])
                pairs.append(f"{target} ({snt}): {ops}")
        aspects.append(" | ".join(pairs) if pairs else None)

    df.loc[valid_idx[:len(lab)], "sentiment_label"] = lab
    df.loc[valid_idx[:len(lab)], "positive_score"] = pos
    df.loc[valid_idx[:len(lab)], "neutral_score"] = neu
    df.loc[valid_idx[:len(lab)], "negative_score"] = neg
    df.loc[valid_idx[:len(lab)], "sentiment_score"] = sc
    df.loc[valid_idx[:len(lab)], "sentences_count"] = nsents
    df.loc[valid_idx[:len(lab)], "aspects"] = aspects

    print(f"✅ Sentiment completed: {len(lab)} reviews processed")

    # Key Phrases
    kp_vals = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        for attempt in range(3):
            try:
                resp = ta.extract_key_phrases(batch, language=lang_param)
                kp_vals.extend([" | ".join(r.key_phrases) if not r.is_error else None for r in resp])
                break
            except Exception as e:
                print(f"⚠️ Error in Key Phrases (attempt {attempt+1}/3): {e}")
                time.sleep(1.2 * (attempt + 1))
    df.loc[valid_idx[:len(kp_vals)], "key_phrases"] = kp_vals
    print(f"✅ Key Phrases completed: {len(kp_vals)} reviews")

    # Entities
    ent_rows = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        for attempt in range(3):
            try:
                resp = ta.recognize_entities(batch, language=lang_param)
                ent_rows.extend(resp)
                break
            except Exception as e:
                print(f"⚠️ Error in Entities (attempt {attempt+1}/3): {e}")
                time.sleep(1.2 * (attempt + 1))
    ents = []
    for r in ent_rows:
        if not r.is_error:
            ents.append(" | ".join([f"{e.text} ({e.category})" for e in r.entities]))
        else:
            ents.append(None)
    df.loc[valid_idx[:len(ents)], "entities"] = ents
    print(f"✅ Entities completed: {len(ents)} reviews")

    # Linked Entities
    linked_vals = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        for attempt in range(3):
            try:
                resp = ta.recognize_linked_entities(batch, language=lang_param)
                linked_vals.extend([" | ".join([e.name for e in r.entities]) if not r.is_error else None for r in resp])
                break
            except Exception as e:
                print(f"⚠️ Error in Linked Entities (attempt {attempt+1}/3): {e}")
                time.sleep(1.2 * (attempt + 1))
    df.loc[valid_idx[:len(linked_vals)], "linked_entities"] = linked_vals
    print(f"✅ Linked Entities completed: {len(linked_vals)} reviews")

    # PII
    pii_vals = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        for attempt in range(3):
            try:
                resp = ta.recognize_pii_entities(batch, language=lang_param)
                pii_vals.extend([" | ".join([f"{e.text} ({e.category})" for e in r.entities]) if not r.is_error else None for r in resp])
                break
            except Exception as e:
                print(f"⚠️ Error in PII (attempt {attempt+1}/3): {e}")
                time.sleep(1.2 * (attempt + 1))
    df.loc[valid_idx[:len(pii_vals)], "pii_entities"] = pii_vals
    print(f"✅ PII completed: {len(pii_vals)} reviews")

    # Metadata
    df["scored_at"] = datetime.now(timezone.utc)

    # Filter out reviews with "Mixed" sentiment
    if "sentiment_label" in df.columns:
        before = len(df)
        df = df[df["sentiment_label"].str.lower() != "mixed"].copy()
        after = len(df)
        print(f"🧹 Filtered reviews with 'Mixed' sentiment: {before - after} of {before}")
    else:
        print("⚠️ Column 'sentiment_label' not found; 'Mixed' filter not applied.")

    # Normalize aspects to thematic categories (two-level model)
    # IMPORTANT: Only reviews with text (processed by Azure) will have aspects
    from .aspect_normalization import normalize_aspects_column
    
    # Initialize aspect columns for all rows (None for reviews without text)
    df["aspect_theme"] = None
    df["aspect_raw"] = None
    
    if "aspects" in df.columns:
        print("🔄 Normalizing aspects to thematic categories...")
        
        # Only normalize aspects for reviews that were actually processed (have text_used)
        # Reviews without text will have None for aspect_theme and aspect_raw
        df = normalize_aspects_column(
            df, 
            aspect_col="aspects", 
            output_theme_col="aspect_theme",
            output_raw_col="aspect_raw"
        )
        
        # Count reviews with actual aspects (non-null aspect_theme)
        reviews_with_aspects = df["aspect_theme"].notna().sum()
        reviews_without_text = len(df) - reviews_with_aspects
        
        # Validate normalization (only for reviews with aspects)
        from .aspect_normalization import validate_thematic_normalization
        validation = validate_thematic_normalization(df, theme_col="aspect_theme")
        if validation["valid"]:
            print(f"✅ Aspect normalization complete: {validation['total_unique_themes']} unique themes")
            print(f"   📊 Reviews with aspects: {reviews_with_aspects} | Reviews without text: {reviews_without_text}")
            if validation.get("generic_categories_found"):
                print(f"⚠️ WARNING: Generic categories found (should not exist): {validation['generic_categories_found']}")
        else:
            print(f"⚠️ Aspect normalization validation: {len(validation['invalid_themes'])} invalid themes found: {validation['invalid_themes']}")
    else:
        print("⚠️ Column 'aspects' not found; skipping aspect normalization.")
        # Keep None values (reviews without text should not have aspects)

    print(f"✅ Enrichment complete: {len(valid_idx)} reviews analyzed with Azure.")
    return df


def normalize_dataframe(df: pd.DataFrame, data_columns: list) -> pd.DataFrame:
    """
    Normalize DataFrame for database insertion.
    
    Converts data types, handles nulls, and ensures consistency.
    
    Args:
        df: DataFrame to normalize
        data_columns: List of expected column names
        
    Returns:
        Normalized DataFrame
    """
    df = df.copy()

    for col in df.select_dtypes(include=["int64"]).columns:
        if df[col].isna().any():
            df[col] = df[col].astype("Int64")

    for col in df.columns:
        if str(df[col].dtype).startswith("datetime64"):
            if df[col].isna().all():
                continue
            s = pd.to_datetime(df[col], errors="coerce", utc=True)
            df[col] = s.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    for col in df.select_dtypes(include=["bool"]).columns:
        df[col] = df[col].astype("boolean")

    for c in data_columns:
        if c not in df.columns:
            df[c] = pd.NA

    for c in {
        "stars", "totalScore", "reviewsCount", "hotelStars", "price",
        "reviewerNumberOfReviews", "likesCount", "response_delay_days",
        "reviewDetailedRating/Location", "reviewDetailedRating/Rooms",
        "reviewDetailedRating/Service", "positive_score", "neutral_score",
        "negative_score", "sentiment_score"
    }:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in {"review_length", "sentences_count"}:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    return df[data_columns]


def union_and_deduplicate(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Union multiple DataFrames and deduplicate by reviewId.
    
    Args:
        dfs: List of DataFrames to union
        
    Returns:
        Deduplicated DataFrame
    """
    df = pd.concat(dfs, ignore_index=True)
    if "hotel_name" not in df.columns:
        df["hotel_name"] = pd.NA
    if "reviewId" not in df.columns:
        raise SystemExit("Dataset does not contain 'reviewId' for deduplication.")
    if "scrapedAt" in df.columns:
        order = pd.to_datetime(df["scrapedAt"], errors="coerce", utc=True)
        df = df.loc[order.sort_values(kind="stable").index].drop_duplicates(
            subset=["reviewId"], keep="last"
        )
    else:
        df = df.drop_duplicates(subset=["reviewId"], keep="first")
    return df.reset_index(drop=True)

