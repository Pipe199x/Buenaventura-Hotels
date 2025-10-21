# src/gold_build.py
import os, time, argparse
from io import BytesIO
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Azure SDK
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

# utils locales
from .utils import ensure_container, df_to_parquet_bytes, get_container
from .config import GOLD_PREFIX

# ===================== Configuración =====================
load_dotenv()
AZ_CONTAINER = os.getenv("AZURE_CONTAINER_NAME", "datasets")
LANG_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT")
LANG_KEY = os.getenv("AZURE_LANGUAGE_KEY")

BATCH_SIZE = 5  # máximo permitido por Azure (importante!)

def get_ta_client() -> TextAnalyticsClient:
    if not LANG_ENDPOINT or not LANG_KEY:
        raise SystemExit("❌ Faltan AZURE_LANGUAGE_ENDPOINT o AZURE_LANGUAGE_KEY en .env")
    return TextAnalyticsClient(endpoint=LANG_ENDPOINT, credential=AzureKeyCredential(LANG_KEY))


# ===================== Cargar SILVER =====================
def load_silver_parquet(hotel_id: str) -> pd.DataFrame:
    container = get_container()
    blob_name = f"silver/{hotel_id}.parquet"
    try:
        data = container.download_blob(blob_name).readall()
    except Exception as e:
        raise SystemExit(f"❌ No pude leer {blob_name}: {e}")
    return pd.read_parquet(BytesIO(data))


def list_silver_hotels() -> list[str]:
    container = get_container()
    hotels = []
    for b in container.list_blobs(name_starts_with="silver/"):
        name = b.name
        if name.endswith(".parquet") and name.count("/") == 1:
            hotels.append(name.split("/", 1)[1].replace(".parquet", ""))
    return sorted(hotels)


# ===================== Enriquecer con Azure =====================
def enrich_with_azure(df_silver: pd.DataFrame, *, language: Optional[str] = "es", mode: str = "cloud") -> pd.DataFrame:
    df = df_silver.copy()

    # --- preparar texto válido ---
    df["text_used"] = df.get("text", "").fillna("").astype(str).str.strip()
    mask_empty = df["text_used"] == ""
    if "textTranslated" in df.columns:
        df.loc[mask_empty, "text_used"] = df.loc[mask_empty, "textTranslated"].fillna("").astype(str).str.strip()

    # --- inicializar columnas ---
    add_cols = [
        "sentiment_label", "positive_score", "neutral_score", "negative_score",
        "sentiment_score", "sentences_count", "aspects",
        "key_phrases", "entities", "pii_entities", "linked_entities",
        "scored_at"
    ]
    for c in add_cols:
        if c not in df.columns:
            df[c] = np.nan

    # --- modo local ---
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

    # --- modo cloud ---
    ta = get_ta_client()
    lang_param = None if language is None else language
    mask_valid = df["text_used"].fillna("").astype(str).str.strip().str.len() > 0
    valid_idx = df.index[mask_valid].tolist()
    texts = df.loc[mask_valid, "text_used"].tolist()

    print(f"🧠 Procesando {len(texts)} reseñas válidas con Azure Language Service...")

    # --- fuerza tipo de columnas (para evitar FutureWarning) ---
    for col in ["sentiment_label", "aspects", "key_phrases"]:
        if col in df.columns:
            df[col] = df[col].astype(object)

    # ---------- 1. Sentiment ----------
    sent_rows = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        for attempt in range(4):
            try:
                resp = ta.analyze_sentiment(batch, language=lang_param, show_opinion_mining=True)
                sent_rows.extend(resp)
                break
            except Exception as e:
                print(f"⚠️ Error en Sentiment (intento {attempt+1}/4): {e}")
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

    print(f"✅ Sentiment completado: {len(lab)} reseñas procesadas")

    # ---------- 2. Key Phrases ----------
    kp_vals = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        for attempt in range(3):
            try:
                resp = ta.extract_key_phrases(batch, language=lang_param)
                kp_vals.extend([" | ".join(r.key_phrases) if not r.is_error else None for r in resp])
                break
            except Exception as e:
                print(f"⚠️ Error en Key Phrases (intento {attempt+1}/3): {e}")
                time.sleep(1.2 * (attempt + 1))
    df.loc[valid_idx[:len(kp_vals)], "key_phrases"] = kp_vals
    print(f"✅ Key Phrases completado: {len(kp_vals)} reseñas")

    # ---------- 3. Entities ----------
    ent_rows = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        for attempt in range(3):
            try:
                resp = ta.recognize_entities(batch, language=lang_param)
                ent_rows.extend(resp)
                break
            except Exception as e:
                print(f"⚠️ Error en Entities (intento {attempt+1}/3): {e}")
                time.sleep(1.2 * (attempt + 1))
    ents = []
    for r in ent_rows:
        if not r.is_error:
            ents.append(" | ".join([f"{e.text} ({e.category})" for e in r.entities]))
        else:
            ents.append(None)
    df.loc[valid_idx[:len(ents)], "entities"] = ents
    print(f"✅ Entities completado: {len(ents)} reseñas")

    # ---------- 4. Linked Entities ----------
    linked_vals = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        for attempt in range(3):
            try:
                resp = ta.recognize_linked_entities(batch, language=lang_param)
                linked_vals.extend([" | ".join([e.name for e in r.entities]) if not r.is_error else None for r in resp])
                break
            except Exception as e:
                print(f"⚠️ Error en Linked Entities (intento {attempt+1}/3): {e}")
                time.sleep(1.2 * (attempt + 1))
    df.loc[valid_idx[:len(linked_vals)], "linked_entities"] = linked_vals
    print(f"✅ Linked Entities completado: {len(linked_vals)} reseñas")

    # ---------- 5. PII ----------
    pii_vals = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        for attempt in range(3):
            try:
                resp = ta.recognize_pii_entities(batch, language=lang_param)
                pii_vals.extend([" | ".join([f"{e.text} ({e.category})" for e in r.entities]) if not r.is_error else None for r in resp])
                break
            except Exception as e:
                print(f"⚠️ Error en PII (intento {attempt+1}/3): {e}")
                time.sleep(1.2 * (attempt + 1))
    df.loc[valid_idx[:len(pii_vals)], "pii_entities"] = pii_vals
    print(f"✅ PII completado: {len(pii_vals)} reseñas")

    # --- metadatos ---
    df["scored_at"] = datetime.now(timezone.utc)

    print(f"✅ Enriquecimiento completo: {len(valid_idx)} reseñas analizadas con Azure.")
    return df


# ===================== Guardar GOLD =====================
def save_gold(df_gold: pd.DataFrame, hotel: str, mode: str = "cloud"):
    file_name = f"{hotel}_GOLD.parquet"
    if mode == "local":
        out_dir = Path("data/gold")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / file_name
        df_gold.to_parquet(out_path, index=False)
        print(f"✅ Gold guardado localmente: {out_path}")
    else:
        container = ensure_container()
        blob_path = f"{GOLD_PREFIX}/{file_name}"
        container.upload_blob(name=blob_path, data=df_to_parquet_bytes(df_gold), overwrite=True)
        print(f"☁️ Subido Gold: {blob_path} ({len(df_gold)} filas)")


# ===================== CLI =====================
def main():
    parser = argparse.ArgumentParser(description="Construir y subir GOLD desde SILVER")
    parser.add_argument("--hotel", required=True, help='ID del hotel, o "all" para procesar todos')
    parser.add_argument("--language", default="es", help='Idioma ("es" recomendado); usa "none" para autodetectar')
    parser.add_argument("--mode", default="cloud", help='"cloud" para Azure o "local" para prueba sin conexión')
    args = parser.parse_args()

    lang = None if args.language.lower() == "none" else args.language
    hotels = list_silver_hotels() if args.hotel.lower() == "all" else [args.hotel]

    if not hotels:
        raise SystemExit("ℹ️ No encontré silver/*.parquet en el contenedor.")

    for h in hotels:
        print(f"\n🟨 Procesando hotel: {h}")
        df_silver = load_silver_parquet(h)
        print(f"SILVER → {df_silver.shape} filas")

        df_gold = enrich_with_azure(df_silver, language=lang, mode=args.mode)
        print(f"GOLD listo → {df_gold.shape} filas")

        save_gold(df_gold, h, mode=args.mode)

    print("\n🎉 Finalizado con éxito.")


if __name__ == "__main__":
    main()
