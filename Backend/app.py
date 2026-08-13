"""Compatibility entrypoint for the migrated SCANU backend.

Run from this directory with:

    python -m uvicorn app:app --host 0.0.0.0 --port 8088
"""

from api.app import app

