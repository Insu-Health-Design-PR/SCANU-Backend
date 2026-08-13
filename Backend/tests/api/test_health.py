"""API route tests."""

from fastapi.testclient import TestClient

from api.app import create_app


def test_health():
    app = create_app()
    client = TestClient(app)
    resp = client.get("/health/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
