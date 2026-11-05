import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
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

st.set_page_config(page_title="AgriVision AI | Plant Health Scanner", page_icon="🌿", layout="centered")

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

init_db()

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@st.cache_resource
def load_model_and_classes():
    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        return None, None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(CLASSES_PATH, "r", encoding="utf-8") as f:
            classes = json.load(f)
        if not isinstance(classes, list) or not classes:
            raise ValueError("classes.json must contain a non-empty JSON list.")
        return model, classes
    except Exception as exc:
        return None, None

MODEL, CLASSES = load_model_and_classes()

def predict_image(image_file):
    if MODEL is None or CLASSES is None:
        raise RuntimeError("The trained model is not installed yet.")

    image = Image.open(image_file).convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    batch = np.expand_dims(array, axis=0)
    batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)

    predictions = MODEL.predict(batch, verbose=0)[0]
    if len(predictions) != len(CLASSES):
        raise RuntimeError("Model output count does not match classes.json.")

    best_index = int(np.argmax(predictions))
    confidence = float(predictions[best_index])

    return CLASSES[best_index], confidence

st.sidebar.title("🌿 AgriVision AI")
page = st.sidebar.radio("Navigation", ["Scanner", "History"])

if page == "Scanner":
    st.title("🌿 AgriVision AI | Plant Health Scanner")
    st.markdown("A smarter first look at plant leaf health. Upload a clear leaf image to receive a model-generated class prediction and confidence score.")
    
    if MODEL is None:
        st.warning("⚠️ Model status: Model setup required. Follow the training instructions to add model artifacts.")
    else:
        st.success("✅ Model loaded successfully.")

    uploaded_file = st.file_uploader("Select a plant leaf image (JPG, JPEG, PNG, WEBP):", type=["png", "jpg", "jpeg", "webp"])

    if uploaded_file is not None:
        if uploaded_file.size > MAX_IMAGE_BYTES:
            st.error("❌ File is too large. Maximum upload size is 5 MB.")
        else:
            try:
                image = Image.open(uploaded_file)
                image.verify()
                uploaded_file.seek(0)
                image = Image.open(uploaded_file)
                
                if image.width * image.height > 20_000_000:
                    st.error("❌ This image is too large. Please choose a smaller image.")
                else:
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                    
                    if st.button("Analyze Image"):
                        with st.spinner("Analyzing image..."):
                            try:
                                label, confidence = predict_image(uploaded_file)
                                
                                formatted_label = label.replace("___", " — ").replace("_", " ")
                                
                                st.subheader("Prediction Result")
                                st.markdown(f"### **{formatted_label}**")
                                st.text("Model confidence score")
                                st.progress(float(confidence))
                                st.write(f"**{confidence * 100:.1f}%**")
                                
                                st.warning("⚠️ Important: This is a model prediction, not a confirmed diagnosis. The confidence score may be poorly calibrated and does not guarantee correctness. Use expert advice before treatment.")
                                
                                filename = secure_filename(uploaded_file.name)[:200] or "uploaded-image"
                                created_file_time = datetime.now(timezone.utc).isoformat(timespec="seconds")
                                with sqlite3.connect(DB_PATH) as conn:
                                    conn.execute(
                                        "INSERT INTO scans (filename, predicted_class, confidence, created_at) VALUES (?, ?, ?, ?)",
                                        (filename, label, confidence, created_file_time),
                                    )
                                    conn.commit()
                            except Exception as e:
                                st.error(f"Prediction failed: {e}")
            except Exception:
                st.error("❌ The selected file is not a valid image.")

elif page == "History":
    st.title("📋 Scan History")
    st.markdown("Metadata from the most recent 100 scans on this deployment.")
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, filename, predicted_class, confidence, created_at "
            "FROM scans ORDER BY id DESC LIMIT 100"
        ).fetchall()
        
    if rows:
        import pandas as pd
        data = []
        for r in rows:
            data.append({
                "#": r["id"],
                "Filename": r["filename"],
                "Predicted Class": r["predicted_class"].replace("___", " — ").replace("_", " "),
                "Confidence": f"{r['confidence'] * 100:.1f}%",
                "Time (UTC)": r["created_at"]
            })
        st.table(pd.DataFrame(data))
    else:
        st.info("No scans yet.")
