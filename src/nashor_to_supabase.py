# scripts/union_upsert_pooler_autoschema.py
from __future__ import annotations

import io
import os
import sys
import time
import argparse
import socket
import math
from typing import List, Dict, Tuple
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
import psycopg
from psycopg import sql

# ==================== ENV ====================
load_dotenv()
AZ_CONN = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZ_CONTAINER = os.getenv("AZURE_CONTAINER", "datasets")
AZ_PREFIX = os.getenv("AZURE_PREFIX", "gold/")
DB_URL = os.getenv("SUPABASE_DB_URL")
TABLE = os.getenv("DEST_TABLE", "public.hotels_gold")

if not AZ_CONN:
    raise SystemExit("Falta AZURE_STORAGE_CONNECTION_STRING")
if not DB_URL:
    raise SystemExit("Falta SUPABASE_DB_URL (URI del POOLER)")


# ==================== Pooler URL check ====================
def ensure_pooler_url(url: str) -> str:
    if not (url.startswith("postgresql://") or url.startswith("postgres://")):
        raise SystemExit("SUPABASE_DB_URL debe iniciar con postgresql://")
    p = urlparse(url)
    host = p.hostname or ""
    port = p.port or 6543
    try:
        socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except Exception as e:
        raise SystemExit(f"No se pudo resolver '{host}:{port}': {e}")
    q = dict(parse_qsl(p.query, keep_blank_values=True))
    if q.get("sslmode") is None:
        q["sslmode"] = "require"
    return urlunparse(p._replace(query=urlencode(q)))


DB_URL = ensure_pooler_url(DB_URL)


# ==================== Azure helpers ====================
def get_container():
    return BlobServiceClient.from_connection_string(AZ_CONN).get_container_client(AZ_CONTAINER)


def list_parquets(container, prefix: str) -> list[str]:
    return [
        b.name
        for b in container.list_blobs(name_starts_with=prefix)
        if b.name.lower().endswith(".parquet")
    ]


def read_parquet(container, blob_name: str) -> pd.DataFrame:
    data = container.download_blob(blob_name).readall()
    return pd.read_parquet(io.BytesIO(data))


# ==================== Dataset columns (fuente) ====================
DATA_COLS: List[str] = [
    "hotel_id", "reviewId", "placeId", "title", "text", "textTranslated", "originalLanguage",
    "reviewOrigin", "publishedAtDate", "year_month", "stars", "totalScore", "reviewsCount",
    "hotelStars", "price", "isLocalGuide", "reviewerNumberOfReviews", "likesCount",
    "responseFromOwnerText", "responseFromOwnerDate", "response_delay_days", "review_length",
    "scrapedAt", "categoryName", "reviewUrl", "url", "reviewContext/Food & drinks",
    "reviewContext/Hotel highlights", "reviewContext/Nearby activities",
    "reviewContext/Noteworthy details", "reviewContext/Rooms", "reviewContext/Safety",
    "reviewContext/Travel group", "reviewContext/Trip type", "reviewContext/Walkability",
    "reviewDetailedRating/Location", "reviewDetailedRating/Rooms", "reviewDetailedRating/Service",
    "text_used", "sentiment_label", "positive_score", "neutral_score", "negative_score",
    "sentiment_score", "sentences_count", "aspects", "key_phrases", "entities", "pii_entities",
    "linked_entities", "scored_at", "hotel_name"
]


# ==================== Normalización ====================
def to_utc_iso(series: pd.Series) -> pd.Series:
    if series.isna().all():
        return series
    s = pd.to_datetime(series, errors="coerce", utc=True)
    return s.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.select_dtypes(include=["int64"]).columns:
        if df[col].isna().any():
            df[col] = df[col].astype("Int64")

    for col in df.columns:
        if str(df[col].dtype).startswith("datetime64"):
            df[col] = to_utc_iso(df[col])

    for col in df.select_dtypes(include=["bool"]).columns:
        df[col] = df[col].astype("boolean")

    for c in DATA_COLS:
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

    return df[DATA_COLS]


# ==================== Unión + dedup ====================
def union_and_dedup(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(dfs, ignore_index=True)
    if "hotel_name" not in df.columns:
        df["hotel_name"] = pd.NA
    if "reviewId" not in df.columns:
        raise SystemExit("El dataset no contiene 'reviewId' para deduplicar.")
    if "scrapedAt" in df.columns:
        order = pd.to_datetime(df["scrapedAt"], errors="coerce", utc=True)
        df = df.loc[order.sort_values(kind="stable").index].drop_duplicates(
            subset=["reviewId"], keep="last"
        )
    else:
        df = df.drop_duplicates(subset=["reviewId"], keep="first")
    return df.reset_index(drop=True)


# ==================== Schema introspection ====================
def get_table_schema(conn: psycopg.Connection, table: str) -> Tuple[List[str], str]:
    schema, name = table.split(".") if "." in table else ("public", table)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema=%s AND table_name=%s
            ORDER BY ordinal_position;
            """,
            (schema, name),
        )
        cols = [r[0] for r in cur.fetchall()]

        cur.execute(
            """
            SELECT a.attname
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid
            AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = %s::regclass AND i.indisprimary;
            """,
            (f"{schema}.{name}",),
        )
        pk_cols = [r[0] for r in cur.fetchall()]

    if not cols:
        raise SystemExit(f"No encontré columnas para {table}. ¿La tabla existe?")
    if not pk_cols:
        raise SystemExit(f"La tabla {table} no tiene PK. Define una PK (esperada: reviewid).")
    if len(pk_cols) != 1:
        raise SystemExit(f"PK compuesta no soportada automáticamente: {pk_cols}")

    return cols, pk_cols[0]


def build_column_mapping(real_cols: List[str]) -> Dict[str, str]:
    real_set = set(real_cols)
    mapping: Dict[str, str] = {}
    missing: List[str] = []

    for c in DATA_COLS:
        if c in real_set:
            mapping[c] = c
        elif "/" in c:
            if c in real_set:
                mapping[c] = c
            elif c.lower() in real_set:
                mapping[c] = c.lower()
            else:
                missing.append(c)
        else:
            lc = c.lower()
            if lc in real_set:
                mapping[c] = lc
            else:
                missing.append(c)

    if missing:
        print("⚠️ Columnas del dataset que NO existen en la tabla y se ignorarán:")
        for m in missing:
            print("   -", m)

    return mapping


# ==================== SQL build ====================
def q(name: str):
    return sql.Identifier(name)


def build_stmt(
    table: str, mapping: Dict[str, str], pk_col: str
) -> Tuple[sql.Composed, List[str], List[str]]:
    data_cols = [c for c in DATA_COLS if c in mapping]
    db_cols = [mapping[c] for c in data_cols]
    cols_ident = [q(c) for c in db_cols]
    placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in db_cols)

    insert_head = sql.SQL("INSERT INTO {t} ({cols}) VALUES ({vals})").format(
        t=sql.SQL(table),
        cols=sql.SQL(", ").join(cols_ident),
        vals=placeholders,
    )

    update_cols = [c for c in db_cols if c != pk_col]
    set_pairs = sql.SQL(", ").join(
        sql.SQL("{c} = EXCLUDED.{c}").format(c=q(c)) for c in update_cols
    )

    stmt = insert_head + sql.SQL(" ON CONFLICT ({pk}) DO UPDATE SET {setp}").format(
        pk=q(pk_col), setp=set_pairs
    )
    return stmt, data_cols, db_cols


# ==================== Convertidores ====================
def to_python_scalar(v):
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, np.floating):
        fv = float(v)
        return None if (math.isnan(fv) or math.isinf(fv)) else fv
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def df_rows_to_tuples(df: pd.DataFrame, data_cols: List[str]) -> list[tuple]:
    out = []
    for _, row in df.iterrows():
        out.append(tuple(to_python_scalar(row[c]) for c in data_cols))
    return out


def batched(iterable, size: int):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


# ==================== Funcionalidad TRUNCATE ====================
def truncate_table(conn: psycopg.Connection, table: str):
    print(f"⚠️ TRUNCATE: Vaciando la tabla {table} y reiniciando IDs...")
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY;")
    print("✅ TRUNCATE completado.")


# ==================== Upsert por lotes (pooler) ====================
def upsert_batches(
    conn: psycopg.Connection,
    df: pd.DataFrame,
    stmt: sql.Composed,
    data_cols: List[str],
    *,
    batch_size: int = 200,
    retries: int = 3,
):
    if df.empty:
        print("Nada que insertar.")
        return

    total = len(df)
    done = 0
    for i, idxs in enumerate(batched(range(total), batch_size), 1):
        chunk = df.iloc[list(idxs)]
        values = df_rows_to_tuples(chunk, data_cols)
        for att in range(retries):
            try:
                with conn.transaction():
                    with conn.cursor() as cur:
                        cur.executemany(stmt, values)
                done += len(values)
                print(f"   ✓ Lote {i}: {len(values)} filas (acum {done}/{total})")
                break
            except Exception as e:
                transient = any(
                    t in str(e).lower()
                    for t in ("timeout", "too many", "cancel", "deadlock", "connection", "pool")
                )
                if att < retries - 1 and transient:
                    time.sleep(1.2 * (att + 1))
                    continue
                print(f"   ✗ Lote {i} error: {e}")
                break


# ==================== main ====================
def main():
    p = urlparse(DB_URL)
    host, port, db = p.hostname, p.port or 6543, p.path.lstrip("/") or "postgres"
    print(f"🔌 Pooler: {host}:{port}/{db}")
    with psycopg.connect(DB_URL, connect_timeout=10) as conn:
        with conn.cursor() as cur:
            cur.execute("select 1;")
            cur.fetchone()
    print("✅ Conexión pooler OK.")

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="Procesar un solo parquet (nombre exacto)")
    ap.add_argument("--batch", type=int, default=int(os.getenv("BATCH_SIZE", "200")))
    ap.add_argument("--truncate", action="store_true", help="Vacía la tabla antes de la carga (TRUNCATE).")
    args = ap.parse_args()

    container = get_container()
    blobs = [f"{AZ_PREFIX}{args.only}"] if args.only else list_parquets(container, AZ_PREFIX)
    if not blobs:
        raise SystemExit("No se encontraron .parquet en el prefijo indicado.")
    print(f"📦 Archivos detectados: {len(blobs)}")

    dfs = []
    for blob in blobs:
        df = read_parquet(container, blob)
        if "hotel_name" not in df.columns:
            df["hotel_name"] = os.path.basename(blob).replace("_GOLD.parquet", "").replace("_silver", "")
        dfs.append(df)

    print("🧭 Uniendo y deduplicando por reviewId ...")
    df_all = union_and_dedup(dfs)
    print(f"   - Filas únicas: {len(df_all)}")

    df_all = normalize_df(df_all)

    with psycopg.connect(DB_URL, autocommit=True) as conn:
        if args.truncate:
            truncate_table(conn, TABLE)

        real_cols, pk_col = get_table_schema(conn, TABLE)
        print(f"🔎 PK detectada: {pk_col}")
        mapping = build_column_mapping(real_cols)

        if "reviewId" in mapping and mapping["reviewId"] != pk_col:
            print(
                f"⚠️ La PK real es '{pk_col}', pero el dataset usa 'reviewId'→'{mapping['reviewId']}'. "
                f"Se usará ON CONFLICT ({pk_col})."
            )

        stmt, data_cols, db_cols = build_stmt(TABLE, mapping, pk_col)
        print(f"🧩 Columnas insertadas ({len(db_cols)}): {', '.join(db_cols)}")
        upsert_batches(conn, df_all, stmt, data_cols, batch_size=args.batch)

    print("🎉 Carga unificada completada.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error fatal: {e}", file=sys.stderr)
        sys.exit(1)
