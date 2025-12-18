"""Configuration management for the application."""
import os
from dotenv import load_dotenv

load_dotenv()

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "datasets")

# Layer prefixes
RAW_PREFIX    = "bronze"     # Bronze
SILVER_PREFIX = "silver"
GOLD_PREFIX   = "gold"

# Local folder containing Excel files
LOCAL_RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets"))

