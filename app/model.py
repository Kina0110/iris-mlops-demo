import joblib
import os

model_path = "model_registry/latest_model.pkl"
model = joblib.load(model_path)

def predict_species(features: list) -> tuple:
    prediction = model.predict([features])[0]
    species = ["setosa", "versicolor", "virginica"]
    return species[prediction], os.path.basename(model_path)
