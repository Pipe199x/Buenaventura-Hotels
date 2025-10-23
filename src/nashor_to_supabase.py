# src/nashor_to_supabase.py
import os
import numpy as np
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from supabase import create_client, Client

load_dotenv()

# ---------- CONFIGURACIÓN ----------
AZURE_CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER = os.getenv("AZURE_CONTAINER_NAME", "datasets")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")


# ---------- CONEXIÓN ----------
def get_supabase() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise SystemExit("❌ Faltan SUPABASE_URL o SUPABASE_KEY en .env")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------- CARGAR GOLDs DESDE AZURE ----------
def load_all_gold_from_azure() -> pd.DataFrame:
    print("📥 Cargando GOLDs desde Azure...")
    blob_service = BlobServiceClient.from_connection_string(AZURE_CONN_STR)
    container = blob_service.get_container_client(AZURE_CONTAINER)

    blobs = [b.name for b in container.list_blobs(name_starts_with="gold/") if b.name.endswith(".parquet")]
    if not blobs:
        raise SystemExit("❌ No se encontraron archivos GOLD en Azure.")

    dfs = []
    for blob in blobs:
        print(f"  → Leyendo {blob} ...")
        data = container.download_blob(blob).readall()
        df = pd.read_parquet(BytesIO(data))
        df["hotel_name"] = blob.split("/")[-1].replace("_GOLD.parquet", "").replace("_silver", "")
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"✅ Total filas combinadas: {len(df_all)}")
    return df_all


# ---------- DETECCIÓN DE VALORES PROBLEMÁTICOS ----------
def detect_invalid_values(df: pd.DataFrame):
    invalid_cols = []
    for col in df.columns:
        if df[col].dtype in [float, np.float64, np.float32]:
            bad_vals = df[col][~np.isfinite(df[col].fillna(0))].count()
            if bad_vals > 0:
                invalid_cols.append((col, bad_vals))
        elif df[col].dtype.name.startswith("datetime"):
            nulls = df[col].isna().sum()
            if nulls > 0:
                invalid_cols.append((col, nulls))
    if invalid_cols:
        print("\n⚠️ Columnas con valores no válidos detectadas:")
        for c, n in invalid_cols:
            print(f"   • {c}: {n} valores problemáticos")
    else:
        print("✅ No se detectaron valores no serializables.")


# ---------- SANITIZAR ANTES DE SUBIR ----------
def sanitize_for_json(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convertir fechas a ISO8601
    for col in df.select_dtypes(include=["datetime64[ns, UTC]", "datetime64[ns]", "datetimetz"]).columns:
        df[col] = df[col].apply(lambda x: x.isoformat() if pd.notna(x) else None)

    # Reemplazar infinitos por NaN y luego NaN por None
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.where(pd.notnull(df), None)

    # Convertir floats fuera de rango a None
    for col in df.select_dtypes(include=[float]).columns:
        df[col] = df[col].apply(
            lambda x: None if (x is None or not np.isfinite(x) or abs(x) > 1e308) else float(x)
        )

    # 🔹 Convertir cualquier tipo "object" con números ilegales
    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: None if isinstance(x, str) and x.lower() in ["nan", "inf", "-inf"] else x
        )

    return df



# ---------- SUBIR A SUPABASE ----------
def upload_to_supabase(df: pd.DataFrame):
    sb = get_supabase()
    print("☁️ Subiendo a Supabase (tabla: hotels_gold)...")

    detect_invalid_values(df)
    df = sanitize_for_json(df)
    records = df.to_dict(orient="records")

    batch_size = 500
    uploaded = 0

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        print(f"🧾 Subiendo batch {i//batch_size + 1} ({len(batch)} filas)...")
        try:
            sb.table("hotels_gold").insert(batch).execute()
            uploaded += len(batch)
        except Exception as e:
            print(f"⚠️ Error en batch {i//batch_size + 1}: {e}")
            continue

    print(f"✅ Subida completada: {uploaded}/{len(df)} filas insertadas.")


# ---------- MAIN ----------
def main():
    df_all = load_all_gold_from_azure()
    upload_to_supabase(df_all)
    print("\n🎉 Nashor → Supabase completado sin errores JSON.")


if __name__ == "__main__":
    main()
