import os
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()
svc = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
client = svc.get_container_client(os.getenv("AZURE_CONTAINER_NAME", "datasets-raw"))

for b in client.list_blobs(name_starts_with="raw/"):  # cambia a "" si no usas prefijo
    print(b.name)
