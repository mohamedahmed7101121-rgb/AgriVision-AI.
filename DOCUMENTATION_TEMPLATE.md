# AgriVision AI — Project Documentation

> Replace bracketed placeholders with your team's actual details and measured results. Do not invent evaluation metrics or screenshots.

## 1. Abstract
AgriVision AI is a cloud-deployed web application prototype that classifies selected plant leaf images using a transfer-learning image classifier. Users upload an image and receive a predicted dataset class and model confidence score. The application records scan metadata for review.

## 2. Problem statement
Plant leaf symptoms can be difficult for non-experts to identify. This project explores whether a computer-vision model can classify leaf images from a defined set of dataset categories. It is an educational prototype and does not replace expert diagnosis.

## 3. Objectives
- Train and evaluate an image classifier using transfer learning.
- Develop a usable web interface for image upload and prediction.
- Connect the trained model to a Flask backend.
- Record scan metadata.
- Deploy the application to a cloud provider.
- Document testing, limitations, and future work.

## 4. Scope
Supported classes: [list the exact classes].
Dataset source/license: [insert source and license].
Dataset size after filtering: [insert count].
Image conditions: [describe dataset conditions].

## 5. System architecture
Browser → Flask application → image validation/preprocessing → TensorFlow/Keras model → prediction result → database metadata.

## 6. Technologies
Python, TensorFlow/Keras, MobileNetV2, Flask, HTML, CSS, Bootstrap, SQLite/PostgreSQL, GitHub, [cloud provider].

## 7. Methodology
Describe dataset preparation, split strategy, preprocessing, transfer-learning setup, training settings, random seed, and evaluation procedure.

## 8. Evaluation
Report actual held-out test accuracy, macro precision, macro recall, macro F1, confusion matrix, and per-class counts from `artifacts/evaluation.json`. Explain errors and any class imbalance.

## 9. Application features
Describe Home/Scanner, Prediction Result, Scan History, validation errors, and health endpoint.

## 10. Cloud deployment
Provider: [name].
Repository: [URL].
Deployed URL: [URL].
Build command: [actual command].
Start command: [actual command].
Environment variables (names only; never publish secret values): [names].
Deployment evidence: [insert real screenshots].

## 11. Testing
Document valid-image test, invalid file, unsupported extension, oversized image, missing model, `/health`, history, and deployed end-to-end prediction.

## 12. Limitations
- Limited to classes included in training.
- Dataset images may differ from real-world field images.
- Confidence score is not necessarily calibrated.
- No guarantee of diagnosis accuracy.
- Treatment recommendations are not automated medical/agronomic advice.

## 13. Future work
Collect field images, test domain generalization, calibrate confidence, add user accounts with appropriate security, migrate to persistent managed storage, and evaluate explainability methods.

## 14. Team contributions
[Member names/IDs and actual contributions.]

## 15. References
Cite the dataset source, MobileNetV2 paper/documentation, TensorFlow/Keras documentation, Flask documentation, and cloud provider documentation in the citation style requested by the instructor.
