# AgriVision AI

Cloud-deployable plant leaf disease classification web application built with Flask and TensorFlow/Keras.

## Current scope
- Upload a leaf image and run a trained image classifier.
- Display predicted class and model confidence.
- Store scan metadata in SQLite during local development.
- Show recent scan history.
- Includes a training script using MobileNetV2 transfer learning.
- Includes test scaffolding and deployment notes.

> Educational prototype only. It is not a professional agricultural diagnosis tool. Model confidence is not a guarantee of correctness. Do not make pesticide or treatment decisions solely from its output.

## 1. Requirements
- Python 3.10 or 3.11 recommended.
- A laptop/desktop.
- Internet for installing dependencies and obtaining a dataset.
- Google Colab is recommended for model training if local hardware is limited.

## 2. Install and run locally
Create and activate a virtual environment, then install dependencies.

### Windows (PowerShell)
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS/Linux
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Prepare the trained model as described in `training/README.md`. The app will start without a model, but prediction will show a clear setup message.

Run:
```bash
python app.py
```
Open http://127.0.0.1:5000

## 3. Dataset setup
Use PlantVillage or another dataset you have permission to use. The training script expects this structure:
```text
data/plantvillage/
  Tomato___Early_blight/
    image1.jpg
    image2.jpg
  Tomato___healthy/
    image3.jpg
  ...
```
Each class must be a folder containing images. Start with 3–5 classes and ensure each class has enough images. The script splits data into train/validation/test sets with a fixed seed. Review `training/README.md` before training.

Avoid randomly splitting near-duplicate images into separate sets if your dataset contains duplicates or multiple images of the same leaf. The resulting score may be overly optimistic.

## 4. Model artifacts
After training, copy the generated artifacts into `artifacts/`:
- `plant_disease_model.keras`
- `classes.json`

Then run the Flask app. Never commit large model files or private credentials to a public repository without checking repository limits and permissions.

## 5. Tests
Install development dependencies if needed:
```bash
pip install -r requirements-dev.txt
pytest -q
```

## 6. Cloud deployment
See `DEPLOYMENT.md`. For persistent scan history, use a managed PostgreSQL database or a persistent disk supported by your host. A local SQLite file on an ephemeral filesystem may be reset during redeploys/restarts.

## Suggested team split
1. Model training and experiments.
2. Dataset preparation and evaluation.
3. Frontend and user experience.
4. Flask backend and database.
5. Integration, deployment, documentation and presentation (with everyone reviewing).

## Current limitations
- The app expects a trained model to be supplied; model weights are not included in this starter bundle.
- Model performance depends on dataset quality and image conditions.
- A confidence score is not the same as a calibrated probability.
- The initial version does not authenticate users; do not expose private scan history on a public deployment.
