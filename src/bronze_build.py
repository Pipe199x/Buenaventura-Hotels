from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime
from typing import Iterable

from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.core.exceptions import ResourceExistsError  # ← AQUÍ EL CAMBIO
from dotenv import load_dotenv

load_dotenv()

CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER = os.getenv("AZURE_CONTAINER_NAME", "datasets")

PROJ_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATA = PROJ_ROOT / "Datasets"

DATE_PREFIX = datetime.utcnow().strftime("%Y-%m-%d")
BLOB_PREFIX = "bronze/"   # pon "" si no quieres subcarpeta

ALLOWED_EXT = {".xlsx", ".xls"}       # agrega ".csv" si quieres

def ensure_env():
    if not CONN_STR:
        raise SystemExit("❌ Falta AZURE_STORAGE_CONNECTION_STRING en .env")
    if not LOCAL_DATA.exists():
        raise SystemExit(f"❌ No existe la carpeta local {LOCAL_DATA}")

def guess_content_type(path: Path) -> ContentSettings:
    ext = path.suffix.lower()
    if ext in {".xlsx", ".xls"}:
        return ContentSettings(
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    if ext == ".csv":
        return ContentSettings(content_type="text/csv")
    return ContentSettings(content_type="application/octet-stream")

def iter_files(folder: Path, allowed: Iterable[str]) -> Iterable[Path]:
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in allowed:
            yield p

def main():
    ensure_env()
    svc = BlobServiceClient.from_connection_string(CONN_STR)
    container = svc.get_container_client(CONTAINER)

    try:
        container.create_container()
        print(f"🆕 Contenedor creado: {CONTAINER}")
    except ResourceExistsError:
        pass

    uploaded = 0
    for path in iter_files(LOCAL_DATA, ALLOWED_EXT):
        blob_name = f"{BLOB_PREFIX}{path.name}"
        with open(path, "rb") as fh:
            container.upload_blob(
                name=blob_name,
                data=fh,
                overwrite=True,
                content_settings=guess_content_type(path),
            )
        uploaded += 1
        print(f"✅ Subido: {path.name}  →  {blob_name}")

    if uploaded == 0:
        print("ℹ️ No encontré archivos .xlsx/.xls en Datasets/")
    else:
        print(f"🎉 Listo. Archivos subidos: {uploaded}")

if __name__ == "__main__":
    main()
