
# MLOps Demo Project: Feature Checklist

## Core Completed
- [x] Train and save Iris classifier model (`retrain_model.py`)
- [x] Serve model via FastAPI (`/predict` endpoint)
- [x] Expose Prometheus metrics via `/metrics`

---

## Next Priorities

### Model Versioning + Retraining Logic 
- [x] Save model with version/timestamp (`iris_model_v2.pkl`, etc.)
- [x] Update `model.py` to load latest version automatically
- [x] Add version info to `/predict` response
- [x] Add retraining trigger simulation (e.g., cron/manual)

### CI/CD Setup (GitHub Actions)
- [x] Add `black` formatter check in `.github/workflows/ci.yml`
- [x] Add basic test (`test_predict.py`) for input/output validation
- [x] Enable automatic lint/test on push

---

## Monitoring Expansion
- [x] Create `prometheus.yml` scrape config
- [x] Run Prometheus via Docker
- [ ] Add Grafana container and dashboard (optional)
- [ ] Log input/output + latency in app logs

---

## Experiment Tracking
- [ ] Add MLflow to `retrain_model.py`
- [ ] Log params, metrics, and versioned models
- [ ] (Optional) Add MLflow UI to Docker Compose

---

## Feature Store Simulation (Optional)
- [ ] Use pandas to simulate feature ingestion + caching
- [ ] Replace raw API input with lookup from preloaded feature table
- [ ] (Stretch goal) Try `Feast` integration

---

## Deployment (Optional)
- [ ] Deploy FastAPI app to Hugging Face Spaces / Render / GCP
- [ ] Secure endpoint with CORS or API key
- [ ] Document endpoint for external use
