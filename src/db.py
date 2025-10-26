# src/db.py
from __future__ import annotations
import os
from pathlib import Path
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from typing import Final
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

ENV_KEYS: Final = ("DATABASE_URL", "SUPABASE_DB_URL")

def _ensure_ssl(url: str) -> str:
    p = urlparse(url)
    q = dict(parse_qsl(p.query, keep_blank_values=True))
    q.setdefault("sslmode", "require")
    return urlunparse(p._replace(query=urlencode(q)))

def _apply_pooler(url: str) -> str:
    """Forzar pooler si USE_POOLER=true. Why: evitar DNS/IPv6 del host 'db.<ref>.supabase.co'."""
    if os.getenv("USE_POOLER", "").lower() not in ("1", "true", "yes"):
        return url
    host = os.getenv("POOLER_HOST", "").strip()   # ej: aws-1-us-east-2.pooler.supabase.com
    port = os.getenv("POOLER_PORT", "6543").strip()
    if not host:
        raise RuntimeError("USE_POOLER=true pero falta POOLER_HOST en .env")
    p = urlparse(url)
    # OJO: algunos proyectos requieren usuario 'postgres.<tenant>'. Copia el usuario EXACTO del panel.
    return urlunparse(p._replace(netloc=f"{p.username}:{p.password}@{host}:{port}"))

def get_db_url() -> str:
    base = next((os.getenv(k, "").strip() for k in ENV_KEYS if os.getenv(k)), "")
    if not base:
        raise RuntimeError(f"No encontré ninguna de {ENV_KEYS} en el entorno.")
    return _ensure_ssl(_apply_pooler(base))

_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(get_db_url(), pool_pre_ping=True, pool_size=5, max_overflow=5)
    return _engine

def ping() -> str:
    with get_engine().connect() as conn:
        ver = conn.execute(text("select version()")).scalar()
    return f"OK Postgres: {ver}"

if __name__ == "__main__":
    print(ping())
