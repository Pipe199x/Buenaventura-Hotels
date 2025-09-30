from __future__ import annotations
import os, time, argparse
from io import BytesIO
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

# utilidades locales
from .utils import ensure_container, df_to_parquet_bytes, get_container
from .config import GOLD_PREFIX  # ej: "gold"

# ===================== Config & clientes =====================
load_dotenv()

AZ_CONTAINER = os.getenv("AZURE_CONTAINER_NAME", "datasets")
LANG_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT")
LANG_KEY      = os.getenv("AZURE_LANGUAGE_KEY")

def get_ta_client() -> TextAnalyticsClient:
    if not LANG_ENDPOINT or not LANG_KEY:
        raise SystemExit("❌ Faltan AZURE_LANGUAGE_ENDPOINT o AZURE_LANGUAGE_KEY en .env")
    return TextAnalyticsClient(endpoint=LANG_ENDPOINT, credential=AzureKeyCredential(LANG_KEY))

# ===================== Cargar SILVER desde Blob =====================
def load_silver_parquet(hotel_id: str) -> pd.DataFrame:
    """
    Layout simple: silver/{hotel_id}.parquet dentro del contenedor.
    Descarga a memoria y lee con pandas (engine=pyarrow).
    """
    container = get_container()
    blob_name = f"silver/{hotel_id}.parquet"
    try:
        data = container.download_blob(blob_name).readall()
    except Exception as e:
        raise SystemExit(f"❌ No pude leer {blob_name}: {e}")
    return pd.read_parquet(BytesIO(data))  # requiere pyarrow instalado

def list_silver_hotels() -> list[str]:
    """Devuelve ['torre_mar','steven_buenaventura', ...] buscando silver/*.parquet"""
    container = get_container()
    hotels = []
    for b in container.list_blobs(name_starts_with="silver/"):
        name = b.name  # p.ej. "silver/torre_mar.parquet"
        if name.endswith(".parquet") and name.count("/") == 1:
            hotels.append(name.split("/", 1)[1].replace(".parquet", ""))
    return sorted(hotels)

# ===================== Helpers de texto =====================
def choose_text(df: pd.DataFrame) -> pd.Series:
    """
    Elige el texto a puntuar: 'text' si existe/no vacío; si no, 'textTranslated'.
    Devuelve una Serie alineada al índice de df.
    """
    t1 = df.get("text", pd.Series(index=df.index, dtype=object)).fillna("").astype(str).str.strip()
    t2 = df.get("textTranslated", pd.Series(index=df.index, dtype=object)).fillna("").astype(str).str.strip()
    return t1.where(t1 != "", t2)

def batched(seq: List[str], n: int = 25):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

# ===================== Enriquecer con Azure AI Language =====================
def enrich_with_azure(df_silver: pd.DataFrame, *, language: Optional[str] = "es") -> pd.DataFrame:
    """
    Enriquecer SILVER con: sentimiento + opinion mining, key phrases, entidades, PII, linked entities.
    - Fecha: solo 2020-01-01 .. 2025-08-21 (UTC)
    - Procesa TODO texto con longitud >= 1. Texto vacío → se deja NaN en métricas.
    - Asignación segura por índice (evita errores de broadcast).
    """
    ta = get_ta_client()
    df = df_silver.copy()

    # --- FECHAS: convertir SIEMPRE a datetime UTC y filtrar rango ---
    df["publishedAtDate"] = pd.to_datetime(df["publishedAtDate"], errors="coerce", utc=True)
    start = pd.Timestamp("2020-01-01", tz="UTC")
    end   = pd.Timestamp("2025-08-21 23:59:59", tz="UTC")
    df = df.loc[df["publishedAtDate"].between(start, end, inclusive="both")].copy()

    # Texto a usar
    df["text_used"] = choose_text(df)

    # --- PROCESAR TODOS LOS TEXTOS (>=1 char). Vacíos quedan con NaN en sentimiento ---
    mask_proc = df["text_used"].fillna("").str.len() >= 1
    idx  = df.index[mask_proc].tolist()
    texts = df.loc[idx, "text_used"].tolist()

    # Inicializa columnas destino
    add_cols = dict(
        detected_language=("auto" if language is None else language),
        sentiment_label=None, positive_score=np.nan, neutral_score=np.nan, negative_score=np.nan,
        sentiment_score=np.nan, sentences_count=0, aspects=None,
        key_phrases=None,
        entities=None, pii_entities=None, linked_entities=None,
        sentiment_model="azure-textanalytics", model_version="v3", scored_at=datetime.now(timezone.utc)
    )
    for k, v in add_cols.items():
        df[k] = v

    # Parámetro de idioma para el SDK (None = autodetect)
    lang_param = None if (language is None) else language

    # ---------- 1) Sentiment + Opinion Mining ----------
    sent_rows = []
    for batch in batched(texts):
        for attempt in range(4):
            try:
                resp = ta.analyze_sentiment(batch, language=lang_param, show_opinion_mining=True)
                sent_rows.extend(resp); break
            except Exception:
                time.sleep(1.5 * (attempt + 1))

    lab, pos, neu, neg, sc, nsents, aspects = [], [], [], [], [], [], []
    for r in sent_rows:
        if getattr(r, "is_error", False):
            lab.append(None); pos.append(np.nan); neu.append(np.nan); neg.append(np.nan); sc.append(np.nan); nsents.append(0); aspects.append(None)
            continue
        lab.append(str(r.sentiment))
        pos.append(float(r.confidence_scores.positive))
        neu.append(float(r.confidence_scores.neutral))
        neg.append(float(r.confidence_scores.negative))
        sc.append((pos[-1] - neg[-1] + 1) / 2)  # [0,1]
        nsents.append(len(r.sentences))
        pairs = []
        for s in r.sentences:
            for mo in getattr(s, "mined_opinions", []):
                target = mo.target.text
                snt    = mo.target.sentiment
                ops    = ", ".join([a.text for a in mo.assessments])
                pairs.append(f"{target} ({snt}): {ops}")
        aspects.append(" | ".join(pairs) if pairs else None)

    # Asignación segura por índice (puede que Azure devuelva < len(idx))
    sent_df = pd.DataFrame({
        "sentiment_label": lab,
        "positive_score":  pos,
        "neutral_score":   neu,
        "negative_score":  neg,
        "sentiment_score": sc,
        "sentences_count": nsents,
        "aspects":         aspects
    }, index=pd.Index(idx[:len(lab)], name="idx"))
    for col in sent_df.columns:
        df.loc[sent_df.index, col] = sent_df[col].values

    # ---------- 2) Key Phrases ----------
    kp_rows = []
    for batch in batched(texts):
        for attempt in range(3):
            try:
                kp_rows.extend(ta.extract_key_phrases(batch, language=lang_param)); break
            except Exception:
                time.sleep(1.2 * (attempt + 1))
    kp_idx = idx[:len(kp_rows)]
    kp_vals = [", ".join(r.key_phrases) if not getattr(r, "is_error", False) else None for r in kp_rows]
    df.loc[kp_idx, "key_phrases"] = kp_vals

    # ---------- 3) Named Entities (NER) ----------
    ent_rows = []
    for batch in batched(texts):
        for attempt in range(3):
            try:
                ent_rows.extend(ta.recognize_entities(batch, language=lang_param)); break
            except Exception:
                time.sleep(1.2 * (attempt + 1))
    ent_idx = idx[:len(ent_rows)]
    ent_vals = [
        ", ".join([f"{e.text}/{e.category}" for e in r.entities]) if not getattr(r, "is_error", False) else None
        for r in ent_rows
    ]
    df.loc[ent_idx, "entities"] = ent_vals

    # ---------- 4) PII ----------
    pii_rows = []
    for batch in batched(texts):
        for attempt in range(3):
            try:
                pii_rows.extend(ta.recognize_pii_entities(batch, language=lang_param)); break
            except Exception:
                time.sleep(1.2 * (attempt + 1))
    pii_idx = idx[:len(pii_rows)]
    pii_vals = [
        ", ".join([f"{e.text}/{e.category}" for e in r.entities]) if not getattr(r, "is_error", False) else None
        for r in pii_rows
    ]
    df.loc[pii_idx, "pii_entities"] = pii_vals

    # ---------- 5) Linked Entities ----------
    link_rows = []
    for batch in batched(texts):
        for attempt in range(3):
            try:
                link_rows.extend(ta.recognize_linked_entities(batch, language=lang_param)); break
            except Exception:
                time.sleep(1.2 * (attempt + 1))
    link_idx = idx[:len(link_rows)]
    link_vals = [
        ", ".join([f"{le.name}→{le.url}" for le in r.entities]) if not getattr(r, "is_error", False) else None
        for r in link_rows
    ]
    df.loc[link_idx, "linked_entities"] = link_vals

    # metadatos del scoring
    df["sentiment_model"] = "azure-textanalytics"
    df["model_version"]   = "v3"
    df["scored_at"]       = datetime.now(timezone.utc)

    return df

# ===================== Subida GOLD =====================
def upload_gold_parquet(df_gold: pd.DataFrame, hotel_id: str):
    container = ensure_container()
    blob_path = f"{GOLD_PREFIX}/{hotel_id}.parquet"
    container.upload_blob(name=blob_path, data=df_to_parquet_bytes(df_gold), overwrite=True)
    print(f"✅ Subido Gold: {blob_path} ({len(df_gold)} filas)")

# ===================== CLI =====================
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
