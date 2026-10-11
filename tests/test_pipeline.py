import sys

sys.path.insert(0, ".")
from src.ingest import _chunk_text  # noqa: E402


def test_chunk_text_respects_overlap():
    text = "a" * 2000
    chunks = _chunk_text(text, chunk_size=800, overlap=100)

    # Every chunk except possibly the last should be exactly chunk_size long.
    for chunk in chunks[:-1]:
        assert len(chunk) == 800

    # Consecutive chunks should share the overlapping tail/head.
    assert chunks[0][-100:] == chunks[1][:100]


def test_chunk_text_handles_empty_string():
    assert _chunk_text("") == []


def test_chunk_text_handles_short_string():
    # A string shorter than chunk_size should come back as a single chunk.
    chunks = _chunk_text("short page text", chunk_size=800, overlap=100)
    assert chunks == ["short page text"]


def test_api_health():
    from fastapi.testclient import TestClient

    from app.api import app

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
