# src/silver_build.py
from __future__ import annotations
import os, io
from pathlib import Path
from datetime import timezone
import pandas as pd
import numpy as np

from .config import LOCAL_RAW_DIR, SILVER_PREFIX
from .utils import ensure_container, iter_files, df_to_parquet_bytes

# ---------- 1) TRANSFORMACIÓN: construir Silver ----------
def make_silver(df: pd.DataFrame, hotel_id: str) -> pd.DataFrame:
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

    # Fechas a UTC
    for c in ["publishedAtDate","responseFromOwnerDate","scrapedAt"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)

    # Numéricos
    for c in ["stars","reviewsCount","hotelStars","reviewerNumberOfReviews","likesCount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in det:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Booleano consistente
    if "isLocalGuide" in df.columns:
        df["isLocalGuide"] = df["isLocalGuide"].map(
            {True: True, False: False, "true": True, "false": False, "True": True, "False": False}
        )

    # Dedupe por reviewId (quedarse con el más reciente)
    if "reviewId" in df.columns:
        sort_keys = [c for c in ["scrapedAt","publishedAtDate"] if c in df.columns]
        if sort_keys:
            df = df.sort_values(sort_keys).drop_duplicates(subset=["reviewId"], keep="last")

    # Derivados útiles
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
    return df[ordered]

# ---------- 2) SUBIDA A AZURE: silver/... en Parquet ----------

def upload_silver_parquet(df_silver: pd.DataFrame, hotel_id: str):
    from .utils import df_to_parquet_bytes, ensure_container
    container = ensure_container()

    blob_path = f"silver/{hotel_id}.parquet"      # << simple
    container.upload_blob(name=blob_path,
                          data=df_to_parquet_bytes(df_silver),
                          overwrite=True)
    print(f"✅ Subido Silver: {blob_path} ({len(df_silver)} filas)")


# ---------- 3) PUNTO DE ENTRADA: lee Excels locales y procesa ----------
def run_for_hotel(excel_path: Path, hotel_id: str, sheet_name="Data"):
    print(f"→ Leyendo {excel_path.name} (sheet={sheet_name})")
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df_silver = make_silver(df, hotel_id=hotel_id)
    print("Silver shape:", df_silver.shape)
    upload_silver_parquet(df_silver, hotel_id)

def main():
    # Mapea archivo → hotel_id (ajústalo a tus archivos)
    mapping = {
        "Hotel_Torre_Mar.xlsx":          "torre_mar",
        "Hotel_Steven_Buenaventura.xlsx":"steven_buenaventura",
        "Hotel_Maguipi.xlsx":            "maguipi",
        "Hotel_Cordillera_Buenaventura.xlsx": "cordillera",
        "Cosmos_Pacifico_Hotel.xlsx":    "cosmos_pacifico",
    }

    for fname, hotel_id in mapping.items():
        excel = Path(LOCAL_RAW_DIR) / fname
        if excel.exists():
            run_for_hotel(excel, hotel_id)
        else:
            print(f"⚠ No encontré {excel}")

if __name__ == "__main__":
    main()
