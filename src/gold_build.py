from __future__ import annotations
import os, time, argparse
from io import BytesIO
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyarrow import fs

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

# --- utilidades locales ---
from .utils import ensure_container, df_to_parquet_bytes, get_container
from .config import GOLD_PREFIX  # p.ej. "gold"

# -------------------- Config & clientes --------------------
load_dotenv()

AZ_CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZ_CONTAINER = os.getenv("AZURE_CONTAINER_NAME", "datasets")

LANG_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT")
LANG_KEY      = os.getenv("AZURE_LANGUAGE_KEY")

def get_ta_client() -> TextAnalyticsClient:
    if not LANG_ENDPOINT or not LANG_KEY:
        raise SystemExit("❌ Faltan AZURE_LANGUAGE_ENDPOINT o AZURE_LANGUAGE_KEY en .env")
    return TextAnalyticsClient(endpoint=LANG_ENDPOINT, credential=AzureKeyCredential(LANG_KEY))

def get_azure_fs() -> fs.AzureFileSystem:
    # PyArrow Azure FS permite leer Parquet directamente desde Blob usando la connection string
    # (lectura directa; no hace falta descargar a disco)  ← docs Arrow FS
    return fs.AzureFileSystem(connection_string=AZ_CONN_STR)

# -------------------- Carga Silver --------------------
def load_silver_parquet(hotel_id: str) -> pd.DataFrame:
    """
    Layout simple: silver/{hotel_id}.parquet dentro del contenedor.
    Descarga el blob a memoria y lo lee con pandas.
    """
    container = get_container()
    blob_name = f"silver/{hotel_id}.parquet"
    try:
        data = container.download_blob(blob_name).readall()
    except Exception as e:
        raise SystemExit(f"❌ No pude leer {blob_name}: {e}")
    return pd.read_parquet(BytesIO(data))  # engine=pyarrow

# -------------------- Helpers TA --------------------
def choose_text(df: pd.DataFrame) -> pd.Series:
    t1 = df.get("text", pd.Series(index=df.index)).fillna("").astype(str).str.strip()
    t2 = df.get("textTranslated", pd.Series(index=df.index)).fillna("").astype(str).str.strip()
    return np.where(t1 != "", t1, t2)

def batched(seq: List[str], n: int = 25):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def list_silver_hotels() -> list[str]:
    """
    Devuelve ['torre_mar','steven_buenaventura', ...] buscando silver/*.parquet
    """
    container = get_container()
    hotels = []
    for b in container.list_blobs(name_starts_with="silver/"):
        name = b.name  # p.ej. "silver/torre_mar.parquet"
        if name.endswith(".parquet") and name.count("/") == 1:
            hotels.append(name.split("/", 1)[1].replace(".parquet", ""))
    return sorted(hotels)

# -------------------- Enriquecer con Azure AI Language --------------------
def enrich_with_azure(df_silver: pd.DataFrame, *, language: Optional[str] = "es") -> pd.DataFrame:
    """
    language="es" recomendado si la mayoría está en español (más rápido y barato).
    language=None → autodetect (ligeramente más lento).
    """
    ta = get_ta_client()
    df = df_silver.copy()

    df["text_used"] = choose_text(df)
    mask = df["text_used"].fillna("").str.len() > 3
    idx  = df.index[mask].tolist()
    texts = df.loc[idx, "text_used"].tolist()

    # inicializa columnas destino
    add_cols = dict(
        detected_language=(language or "auto"),
        sentiment_label=None, positive_score=np.nan, neutral_score=np.nan, negative_score=np.nan,
        sentiment_score=np.nan, sentences_count=0, aspects=None,
        key_phrases=None,
        entities=None, pii_entities=None,
        linked_entities=None,
        sentiment_model="azure-textanalytics", model_version="v3", scored_at=datetime.now(timezone.utc)
    )
    for k,v in add_cols.items():
        df[k] = v

    # 1) Sentiment + Opinion Mining (targets/opinions)
    # API oficial: analyze_sentiment + show_opinion_mining=True :contentReference[oaicite:1]{index=1}
    sent_rows = []
    for batch in batched(texts):
        for attempt in range(4):
            try:
                resp = ta.analyze_sentiment(batch, language=(language or "en"), show_opinion_mining=True)
                sent_rows.extend(resp); break
            except Exception:
                time.sleep(1.5 * (attempt + 1))

    lab,pos,neu,neg,sc,nsents,aspects = [],[],[],[],[],[],[]
    for r in sent_rows:
        if getattr(r, "is_error", False):
            lab.append(None); pos.append(np.nan); neu.append(np.nan); neg.append(np.nan); sc.append(np.nan); nsents.append(0); aspects.append(None)
            continue
        lab.append(str(r.sentiment))
        pos.append(float(r.confidence_scores.positive))
        neu.append(float(r.confidence_scores.neutral))
        neg.append(float(r.confidence_scores.negative))
        sc.append((pos[-1]-neg[-1]+1)/2)       # resumen en [0,1]
        nsents.append(len(r.sentences))
        pairs=[]
        for s in r.sentences:
            for mo in getattr(s, "mined_opinions", []):
                target = mo.target.text
                snt    = mo.target.sentiment
                ops    = ", ".join([a.text for a in mo.assessments])
                pairs.append(f"{target} ({snt}): {ops}")
        aspects.append(" | ".join(pairs) if pairs else None)

    df.loc[idx, ["sentiment_label","positive_score","neutral_score","negative_score","sentiment_score","sentences_count","aspects"]] = \
        np.column_stack([lab,pos,neu,neg,sc,nsents,aspects])

    # 2) Key Phrases (temas principales del texto) :contentReference[oaicite:2]{index=2}
    kp_rows=[]
    for batch in batched(texts):
        kp_rows.extend(ta.extract_key_phrases(batch, language=(language or "en")))
    df.loc[idx, "key_phrases"] = [", ".join(r.key_phrases) if not r.is_error else None for r in kp_rows]

    # 3) Named Entities (NER)  :contentReference[oaicite:3]{index=3}
    ent_rows=[]
    for batch in batched(texts):
        ent_rows.extend(ta.recognize_entities(batch, language=(language or "en")))
    df.loc[idx, "entities"] = [
        ", ".join([f"{e.text}/{e.category}" for e in r.entities]) if not r.is_error else None
        for r in ent_rows
    ]

    # 4) PII (opcional, útil para higiene)  :contentReference[oaicite:4]{index=4}
    pii_rows=[]
    for batch in batched(texts):
        pii_rows.extend(ta.recognize_pii_entities(batch, language=(language or "en")))
    df.loc[idx, "pii_entities"] = [
        ", ".join([f"{e.text}/{e.category}" for e in r.entities]) if not r.is_error else None
        for r in pii_rows
    ]

    # 5) Linked Entities (enlaces a Wikipedia/Bing)  :contentReference[oaicite:5]{index=5}
    link_rows=[]
    for batch in batched(texts):
        link_rows.extend(ta.recognize_linked_entities(batch, language=(language or "en")))
    df.loc[idx, "linked_entities"] = [
        ", ".join([f"{le.name}→{le.url}" for le in r.entities]) if not r.is_error else None
        for r in link_rows
    ]

    # metadatos
    df["sentiment_model"] = "azure-textanalytics"
    df["model_version"]   = "v3"
    df["scored_at"]       = datetime.now(timezone.utc)

    return df

# -------------------- Subida Gold --------------------
def upload_gold_parquet(df_gold: pd.DataFrame, hotel_id: str):
    container = ensure_container()
    blob_path = f"{GOLD_PREFIX}/{hotel_id}.parquet"
    container.upload_blob(name=blob_path, data=df_to_parquet_bytes(df_gold), overwrite=True)
    print(f"✅ Subido Gold: {blob_path} ({len(df_gold)} filas)")

# -------------------- CLI --------------------
def main():
    parser = argparse.ArgumentParser(description="Construir y subir GOLD desde SILVER")
    parser.add_argument("--hotel", required=True, help='ID del hotel, o "all" para procesar todos')
    parser.add_argument("--language", default="es", help='Idioma ("es" recomendado); usa "none" para autodetectar')
    args = parser.parse_args()

    lang = None if args.language.lower() == "none" else args.language
    hotels = list_silver_hotels() if args.hotel.lower() == "all" else [args.hotel]

    if not hotels:
        raise SystemExit("ℹ️ No encontré silver/*.parquet en el contenedor.")

    for h in hotels:
        print(f"\n🟨 Leyendo SILVER de blob: {h}")
        df_silver = load_silver_parquet(h)
        print("SILVER:", df_silver.shape)

        print("🟩 Enriqueciendo con Azure AI Language …")
        df_gold = enrich_with_azure(df_silver, language=lang)
        print("GOLD:", df_gold.shape)

        print("⬆ Subiendo GOLD a blob …")
        upload_gold_parquet(df_gold, h)

    print("\n🎉 Terminado.")


if __name__ == "__main__":
    main()
