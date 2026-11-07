"""Shared MongoDB configuration for all backend services."""

import os
from pathlib import Path

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env")

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "empathy_learning_fresh")


def mongo_settings() -> tuple[str, str]:
    """Return the configured URI and database name without logging secrets."""
    return MONGODB_URI, MONGODB_DATABASE
