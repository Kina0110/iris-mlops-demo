# iris-mlops-demo

An end-to-end MLOps demo that trains, serves, and monitors a machine learning model using FastAPI, Docker, and Prometheus. Built to demonstrate core capabilities like versioned model retraining, live inference, and traceability.

---

## Project Overview

This project includes:

- Iris classifier using `scikit-learn`
- FastAPI server for predictions
- Model retraining with version tracking
- Monitoring endpoint for Prometheus
- Docker support (build/run locally)
- Designed for extensions (CI/CD, MLflow, cloud deployment)

---

## Tech Stack

- Python 3.10+
- FastAPI
- scikit-learn
- joblib
- Docker
- prometheus-fastapi-instrumentator

---

## How It Works

### 1. Model Training

Trains a RandomForest model on the Iris dataset and saves:

- `model_registry/iris_model_<timestamp>.pkl`
- `model_registry/latest_model.pkl`
- `model_registry/version.txt` (used to report the model version in the API)

To run:
```bash
python retrain_model.py
```

---

### 2. API Serving

Run the API locally:
```bash
uvicorn app.main:app --reload
```

Swagger UI:
```
http://127.0.0.1:8000/docs
```

---

### 3. Make a Prediction

POST `/predict` with:
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Sample response:
```json
{
  "prediction": "setosa",
  "model_version": "iris_model_20250508_050812.pkl"
}
```

---

### 4. Trigger a Retrain (Optional)

POST `/retrain` to:
- Train a new model
- Update `latest_model.pkl`
- Update `version.txt`

---

## File Structure

```
model_registry/
├── iris_model_<timestamp>.pkl   # Versioned model
├── latest_model.pkl             # Used by API
└── version.txt                  # Used for traceability
```

---

## Notes

If `version.txt` is missing (e.g., fresh clone), run:
```bash
python retrain_model.py
```

Prometheus metrics are available at:
```
GET /metrics
```

---

## Monitoring with Prometheus

This project exposes built-in metrics at:

```
GET /metrics
```

To enable full monitoring:
1. Make sure the FastAPI app is running on port 8000:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0
   ```

2. Start Prometheus using Docker:
   ```bash
   docker compose up
   ```

Prometheus will be available at:  
```
http://localhost:9090
```

From the dashboard, go to **Status > Targets** to verify scraping is active. You can also query:
- `http_requests_total`
- `http_request_duration_seconds_sum`
- `http_requests_total{status="500"}` for errors

---

## Render Deployment (Public)

This project is also deployed and publicly accessible at:
```
https://iris-mlops-demo.onrender.com/docs
```

#### Example predict request using `feature_id` (from feature store):
```bash
curl -X POST https://iris-mlops-demo.onrender.com/predict \
-H "Content-Type: application/json" \
-d '{"feature_id": 1}'
```

#### Example predict request using raw inputs:
```bash
curl -X POST https://iris-mlops-demo.onrender.com/predict \
-H "Content-Type: application/json" \
-d '{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}'
```

#### Trigger retraining:
```bash
curl -X POST https://iris-mlops-demo.onrender.com/retrain
```

#### Get Prometheus metrics:
```
https://iris-mlops-demo.onrender.com/metrics
```

---

## License

MIT — feel free to use or adapt.
