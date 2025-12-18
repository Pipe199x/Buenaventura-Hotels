"""Backward compatibility: re-export from infrastructure layer."""
from .infrastructure.config import (
    AZURE_STORAGE_CONNECTION_STRING,
    AZURE_CONTAINER_NAME,
    RAW_PREFIX,
    SILVER_PREFIX,
    GOLD_PREFIX,
    LOCAL_RAW_DIR,
)

__all__ = [
    "AZURE_STORAGE_CONNECTION_STRING",
    "AZURE_CONTAINER_NAME",
    "RAW_PREFIX",
    "SILVER_PREFIX",
    "GOLD_PREFIX",
    "LOCAL_RAW_DIR",
]
