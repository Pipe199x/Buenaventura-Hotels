import os
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Cargar .env
load_dotenv()

# Conexión al storage
CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER = os.getenv("AZURE_CONTAINER_NAME", "datasets")

svc = BlobServiceClient.from_connection_string(CONN_STR)
client = svc.get_container_client(CONTAINER)

# Listar blobs
print(f"📂 Contenedor: {CONTAINER}")
blobs = client.list_blobs()

found = False
for blob in blobs:
    print(" -", blob.name)
    found = True

if not found:
    print("⚠️ No se encontraron blobs en este contenedor.")
