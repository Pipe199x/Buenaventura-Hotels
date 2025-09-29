from typing import Iterable
from pathlib import Path
import io

from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.core.exceptions import ResourceExistsError

from .config import AZURE_STORAGE_CONNECTION_STRING, AZURE_CONTAINER_NAME

def get_container():
    svc = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    return svc.get_container_client(AZURE_CONTAINER_NAME)

def ensure_container():
    c = get_container()
    try:
        c.create_container()
    except ResourceExistsError:
        pass
    return c

def iter_files(folder: Path, allowed: Iterable[str]):
    for p in sorted(Path(folder).rglob("*")):
        if p.is_file() and p.suffix.lower() in allowed:
            yield p

def guess_content_type(path: Path) -> ContentSettings:
    ext = path.suffix.lower()
    if ext in {".xlsx", ".xls"}:
        return ContentSettings(content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    if ext == ".csv":
        return ContentSettings(content_type="text/csv")
    if ext == ".parquet":
        return ContentSettings(content_type="application/octet-stream")
    return ContentSettings(content_type="application/octet-stream")

def df_to_parquet_bytes(df):
    import pandas as pd
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)  # requiere pyarrow
    buf.seek(0)
    return buf
