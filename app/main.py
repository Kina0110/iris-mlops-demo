from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model import predict_species
from prometheus_fastapi_instrumentator import Instrumentator
import subprocess

app = FastAPI()
instrumentator = Instrumentator().instrument(app).expose(app)


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def root():
    return {"message": "Iris Classifier API is running."}


@app.post("/predict")
def predict(iris: IrisFeatures):
    try:
        features = [
            iris.sepal_length,
            iris.sepal_width,
            iris.petal_length,
            iris.petal_width,
        ]
        species, model_version = predict_species(features)
        return {"prediction": species, "model_version": model_version}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrain")
def retrain_model():
    try:
        result = subprocess.run(
            ["python", "retrain_model.py"], capture_output=True, text=True
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=result.stderr)
        return {"message": "Model retrained successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
