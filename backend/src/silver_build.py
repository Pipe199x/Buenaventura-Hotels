# src/silver_build.py
from __future__ import annotations
import os, io
from pathlib import Path
from datetime import timezone
import pandas as pd
import numpy as np

from .infrastructure.config import LOCAL_RAW_DIR, SILVER_PREFIX
from .infrastructure.blob_storage import ensure_container, df_to_parquet_bytes
from .domain.transformations import make_silver

# Re-export for backward compatibility
__all__ = ["make_silver", "upload_silver_parquet", "run_for_hotel", "main"]

# ---------- 2) UPLOAD TO AZURE: silver/... as Parquet ----------
def upload_silver_parquet(df_silver: pd.DataFrame, hotel_id: str):
    container = ensure_container()
    blob_path = f"silver/{hotel_id}_silver.parquet"
    container.upload_blob(name=blob_path,
                          data=df_to_parquet_bytes(df_silver),
                          overwrite=True)
    print(f"☁️ Uploaded Silver: {blob_path} ({len(df_silver)} rows)")


# ---------- 3) ENTRY POINT: read local Excel files and process ----------
def run_for_hotel(excel_path: Path, hotel_id: str, sheet_name="Data"):
    """Process a single hotel's Excel file and upload to Silver layer."""
    print(f"→ Reading {excel_path.name} (sheet={sheet_name})")
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df_silver = make_silver(df, hotel_id=hotel_id)
    print("📊 Silver shape:", df_silver.shape)
    upload_silver_parquet(df_silver, hotel_id)

def main():
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
            print(f"⚠ Not found: {excel}")

if __name__ == "__main__":
    main()
