"""Infrastructure layer for external services and configuration."""

from .config import (
    AZURE_STORAGE_CONNECTION_STRING,
    AZURE_CONTAINER_NAME,
    RAW_PREFIX,
    SILVER_PREFIX,
    GOLD_PREFIX,
    LOCAL_RAW_DIR,
)
from .database import get_db_url, get_engine, ping
from .blob_storage import (
    get_container,
    ensure_container,
    iter_files,
    guess_content_type,
    df_to_parquet_bytes,
)

__all__ = [
    "AZURE_STORAGE_CONNECTION_STRING",
    "AZURE_CONTAINER_NAME",
    "RAW_PREFIX",
    "SILVER_PREFIX",
    "GOLD_PREFIX",
    "LOCAL_RAW_DIR",
    "get_db_url",
    "get_engine",
    "ping",
    "get_container",
    "ensure_container",
    "iter_files",
    "guess_content_type",
    "df_to_parquet_bytes",
]

