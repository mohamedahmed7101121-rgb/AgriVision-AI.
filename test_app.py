import io
import os
import tempfile

import pytest

# Use temporary database and a test secret before importing the Flask app.
tmp = tempfile.TemporaryDirectory()
os.environ["DATABASE_PATH"] = os.path.join(tmp.name, "test_scans.db")
os.environ["SECRET_KEY"] = "test-secret"
os.environ["MODEL_PATH"] = os.path.join(tmp.name, "missing_model.keras")
os.environ["CLASSES_PATH"] = os.path.join(tmp.name, "missing_classes.json")

from app import app  # noqa: E402


@pytest.fixture()
def client():
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def test_home_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"AgriVision AI" in response.data


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json["status"] == "ok"
    assert response.json["model_ready"] is False


def test_history_page_loads(client):
    response = client.get("/history")
    assert response.status_code == 200
    assert b"Recent scans" in response.data


def test_missing_upload_is_handled(client):
    response = client.post("/predict", data={}, follow_redirects=True)
    assert response.status_code == 200
    assert b"Choose an image file first" in response.data


def test_unsupported_file_extension_is_rejected(client):
    response = client.post(
        "/predict",
        data={"leaf_image": (io.BytesIO(b"not an image"), "notes.txt")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Supported formats" in response.data
