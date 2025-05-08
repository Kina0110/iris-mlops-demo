import joblib
import numpy as np

# Load the trained model
model = joblib.load("app/iris_model.pkl")

def predict_species(features: list) -> str:
    prediction = model.predict([features])[0]
    species = ["setosa", "versicolor", "virginica"]
    return species[prediction]
