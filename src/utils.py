"""Backward compatibility: re-export from infrastructure layer."""
from .infrastructure.blob_storage import (
    get_container,
    ensure_container,
    iter_files,
    guess_content_type,
    df_to_parquet_bytes,
)

__all__ = [
    "get_container",
    "ensure_container",
    "iter_files",
    "guess_content_type",
    "df_to_parquet_bytes",
]
