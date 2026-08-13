"""ASGI entrypoint for the migrated Layer 8 API."""

import uvicorn

from api.app import app


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8088)
