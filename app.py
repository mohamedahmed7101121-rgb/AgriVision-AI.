import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from flask import Flask, flash, redirect, render_template, request, url_for
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
UPLOAD_DIR = BASE_DIR / "instance" / "uploads"
DB_PATH = Path(os.environ.get("DATABASE_PATH", str(BASE_DIR / "instance" / "scans.db")))
MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(ARTIFACTS_DIR / "plant_disease_model.keras")))
CLASSES_PATH = Path(os.environ.get("CLASSES_PATH", str(ARTIFACTS_DIR / "classes.json")))

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024
IMAGE_SIZE = (224, 224)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-only-change-me")
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model_and_classes():
    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        return None, None
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(CLASSES_PATH, "r", encoding="utf-8") as f:
            classes = json.load(f)
        if not isinstance(classes, list) or not classes:
            raise ValueError("classes.json must contain a non-empty JSON list.")
        return model, classes
    except Exception as exc:
        app.logger.exception("Could not load model artifacts: %s", exc)
        return None, None


MODEL, CLASSES = load_model_and_classes()


def predict_image(image_file):
    if MODEL is None or CLASSES is None:
        raise RuntimeError("The trained model is not installed yet.")

    import tensorflow as tf

    image = Image.open(image_file).convert("RGB").resize(IMAGE_SIZE)

    array = np.asarray(image, dtype=np.float32)
    batch = np.expand_dims(array, axis=0)

    # Apply the same preprocessing used during training
    batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)

    predictions = MODEL.predict(batch, verbose=0)[0]

    if len(predictions) != len(CLASSES):
        raise RuntimeError("Model output count does not match classes.json.")

    best_index = int(np.argmax(predictions))
    confidence = float(predictions[best_index])

    return CLASSES[best_index], confidence


@app.template_filter("percent")
def percent(value):
    return f"{float(value) * 100:.1f}%"


@app.route("/")
def index():
    return render_template("index.html", model_ready=MODEL is not None)


@app.route("/predict", methods=["POST"])
def predict():
    uploaded = request.files.get("leaf_image")
    if not uploaded or not uploaded.filename:
        flash("Choose an image file first.", "error")
        return redirect(url_for("index"))
    if not allowed_file(uploaded.filename):
        flash("Supported formats: JPG, JPEG, PNG, WEBP.", "error")
        return redirect(url_for("index"))
    # Check that the file is actually an image and enforce a pixel limit.
    try:
        uploaded.stream.seek(0)
        image = Image.open(uploaded.stream)
        image.verify()
        uploaded.stream.seek(0)
        image = Image.open(uploaded.stream)
        if image.width * image.height > 20_000_000:
            flash("This image is too large. Please choose a smaller image.", "error")
            return redirect(url_for("index"))
        uploaded.stream.seek(0)
    except (UnidentifiedImageError, OSError, ValueError):
        flash("The selected file is not a valid image.", "error")
        return redirect(url_for("index"))

    try:
        label, confidence = predict_image(uploaded.stream)
    except RuntimeError as exc:
        flash(str(exc), "error")
        return redirect(url_for("index"))
    except Exception:
        app.logger.exception("Prediction failed")
        flash("Prediction failed. Please try another image.", "error")
        return redirect(url_for("index"))

    # Store only metadata; uploaded image is not retained by default.
    filename = secure_filename(uploaded.filename)[:200] or "uploaded-image"
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "INSERT INTO scans (filename, predicted_class, confidence, created_at) VALUES (?, ?, ?, ?)",
            (filename, label, confidence, created_at),
        )
        scan_id = cursor.lastrowid
        conn.commit()
    return render_template(
        "result.html",
        label=label.replace("___", " — ").replace("_", " "),
        raw_label=label,
        confidence=confidence,
        scan_id=scan_id,
    )


@app.route("/history")
def history():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, filename, predicted_class, confidence, created_at "
            "FROM scans ORDER BY id DESC LIMIT 100"
        ).fetchall()
    return render_template("history.html", scans=rows)


@app.route("/health")
def health():
    return {"status": "ok", "model_ready": MODEL is not None}, 200


@app.errorhandler(413)
def too_large(_error):
    flash("File is too large. Maximum upload size is 5 MB.", "error")
    return redirect(url_for("index"))


init_db()

if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
