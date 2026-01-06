# src/gold_build.py
"""Gold layer processing: enrich Silver data with Azure Cognitive Services and save to storage."""
import argparse
from io import BytesIO
from pathlib import Path

import pandas as pd

# Infrastructure and domain imports
from .infrastructure.blob_storage import ensure_container, df_to_parquet_bytes, get_container
from .infrastructure.config import GOLD_PREFIX
from .domain.transformations import enrich_with_azure


# ===================== Load SILVER =====================
def load_silver_parquet(hotel_id: str) -> pd.DataFrame:
    """Load Silver layer parquet file from Azure Blob Storage."""
    container = get_container()
    # Try both naming conventions: with and without _silver suffix
    blob_name = f"silver/{hotel_id}_silver.parquet"
    try:
        data = container.download_blob(blob_name).readall()
    except Exception:
        # Fallback to alternative naming
        blob_name = f"silver/{hotel_id}.parquet"
        try:
            data = container.download_blob(blob_name).readall()
        except Exception as e:
            raise SystemExit(f"❌ Could not read silver/{hotel_id}_silver.parquet or silver/{hotel_id}.parquet: {e}")
    return pd.read_parquet(BytesIO(data))


def list_silver_hotels() -> list[str]:
    """List all hotel IDs available in Silver layer."""
    container = get_container()
    hotels = []
    for b in container.list_blobs(name_starts_with="silver/"):
        name = b.name
        if name.endswith(".parquet") and name.count("/") == 1:
            # Remove _silver suffix if present, and .parquet extension
            hotel_id = name.split("/", 1)[1].replace("_silver.parquet", "").replace(".parquet", "")
            if hotel_id not in hotels:
                hotels.append(hotel_id)
    return sorted(hotels)


# enrich_with_azure is imported from domain.transformations


# ===================== Save GOLD =====================
def save_gold(df_gold: pd.DataFrame, hotel: str, mode: str = "cloud"):
    clean_name = hotel.replace("_silver", "")
    file_name = f"{clean_name}_GOLD.parquet"
    if mode == "local":
        out_dir = Path("data/gold")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / file_name
        df_gold.to_parquet(out_path, index=False)
        print(f"✅ Gold saved locally: {out_path}")
    else:
        container = ensure_container()
        blob_path = f"{GOLD_PREFIX}/{file_name}"
        container.upload_blob(name=blob_path, data=df_to_parquet_bytes(df_gold), overwrite=True)
        print(f"☁️ Uploaded Gold: {blob_path} ({len(df_gold)} rows)")


# ===================== CLI =====================
def main():
    parser = argparse.ArgumentParser(description="Build and upload GOLD from SILVER")
    parser.add_argument("--hotel", required=True, help='Hotel ID, or "all" to process all')
    parser.add_argument("--language", default="es", help='Language ("es" recommended); use "none" for auto-detect')
    parser.add_argument("--mode", default="cloud", help='"cloud" for Azure or "local" for offline testing')
    args = parser.parse_args()

    lang = None if args.language.lower() == "none" else args.language
    hotels = list_silver_hotels() if args.hotel.lower() == "all" else [args.hotel]

    if not hotels:
        raise SystemExit("ℹ️ No silver/*.parquet files found in container.")

    for h in hotels:
        print(f"\n🟨 Processing hotel: {h}")
        df_silver = load_silver_parquet(h)
        print(f"SILVER → {df_silver.shape} rows")

        df_gold = enrich_with_azure(df_silver, language=lang, mode=args.mode)  # Uses domain.transformations.enrich_with_azure
        print(f"GOLD ready → {df_gold.shape} rows")

        save_gold(df_gold, h, mode=args.mode)

    print("\n🎉 Completed successfully.")


if __name__ == "__main__":
    main()
