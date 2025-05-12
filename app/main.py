from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.model import predict_species, get_features_by_id

app = FastAPI()

class PredictRequest(BaseModel):
    feature_id: int = Field(None, description="Feature ID from feature store")
    sepal_length: float = Field(None)
    sepal_width: float = Field(None)
    petal_length: float = Field(None)
    petal_width: float = Field(None)

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        if request.feature_id is not None:
            features = get_features_by_id(request.feature_id)
        elif all([request.sepal_length, request.sepal_width, request.petal_length, request.petal_width]):
            features = [request.sepal_length, request.sepal_width, request.petal_length, request.petal_width]
        else:
            raise HTTPException(status_code=400, detail="Either provide feature_id or all raw features.")
        
        prediction, version = predict_species(features)
        return {"prediction": prediction, "model_version": version}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
