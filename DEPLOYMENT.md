# Cloud deployment checklist (Render example)

The exact dashboard labels and free-plan limits may change; check the provider's current documentation before deployment.

## Before deploying
1. Confirm the app works locally.
2. Confirm `artifacts/plant_disease_model.keras` and `artifacts/classes.json` exist and match.
3. Test `/health` and an end-to-end image prediction.
4. Do not commit API keys, credentials, or private data.
5. Keep model size and memory requirements within the host's limits.

## GitHub
1. Create a repository for the project.
2. Commit source code, templates, static assets, documentation, and dependency files.
3. Do not commit the dataset, `.venv`, secrets, or large model files by default. Check repository file-size limits.
4. If the model is too large for Git, use a documented artifact-storage approach or a provider-supported model download step.

## Render Web Service
1. Create a Web Service from the repository.
2. Build command: `pip install -r requirements.txt`
3. Start command: `gunicorn app:app`
4. Add environment variables:
   - `SECRET_KEY`: a long random secret
   - `MODEL_PATH`: path to the deployed model file if not using the default
   - `CLASSES_PATH`: path to `classes.json` if not using the default
   - `DATABASE_PATH`: persistent path if the host supports persistent storage
5. Deploy and open the service URL.
6. Check `/health`, then test with a valid leaf image.
7. Capture genuine screenshots of the build/deploy logs, successful service, and running website for the presentation.

## Database warning
SQLite is acceptable for local development and demos. On an ephemeral cloud filesystem, SQLite data may disappear on restart/redeploy and may not behave well with multiple workers. For persistent cloud history, use managed PostgreSQL and update the database layer, or use a persistent disk where supported. Never claim persistence unless tested.

## Security and reliability
- Keep the 5 MB upload limit or choose a suitable value.
- Validate file type and actual image content.
- Use a production secret key and debug mode off.
- Do not store uploaded images unless there is a clear need and retention policy.
- Explain that the model is a prototype and predictions can be wrong.
