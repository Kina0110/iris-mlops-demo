# 🧠 iris-mlops-demo

A minimal end-to-end MLOps demo project that trains, serves, and monitors a machine learning model using FastAPI, Docker, and Prometheus.

---

## 🚀 Project Overview

This project demonstrates key MLOps capabilities:

- Training a simple ML model (Iris classification using `scikit-learn`)
- Serving predictions via a REST API using **FastAPI**
- Containerizing the service with **Docker**
- Exposing basic monitoring metrics using **Prometheus**
- Structured for easy extension (e.g., MLflow, CI/CD, cloud deployment)

---

## 📦 Tech Stack

- Python 3.10+
- FastAPI
- scikit-learn
- joblib
- Docker
- prometheus-fastapi-instrumentator

---

## 🔍 How It Works

### 🧪 1. Model Training
Run this to train and save the model locally:
```bash
python retrain_model.py



