"""Environment-based settings."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Config:
    SOFTWARE_ROOT = Path(os.getenv("SOFTWARE_ROOT", Path(__file__).resolve().parents[1]))
    ARTIFACTS_DIR = SOFTWARE_ROOT / os.getenv("ARTIFACTS_DIR", "artifacts")
    CONFIG_DIR = SOFTWARE_ROOT / os.getenv("CONFIG_DIR", "config")
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8088"))
    MODEL_PROFILES_PATH = Path(
        os.getenv("MODEL_PROFILES_PATH", CONFIG_DIR / "profiles" / "model_profiles.json")
    )
    RATELIMIT_STORAGE_URI = os.getenv("RATELIMIT_STORAGE_URI", "memory://")
