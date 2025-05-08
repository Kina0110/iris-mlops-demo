import joblib
import os

model_path = "model_registry/latest_model.pkl"
model = joblib.load(model_path)

def predict_species(features: list) -> tuple:
    prediction = model.predict([features])[0]
    species = ["setosa", "versicolor", "virginica"]

    version_file = "model_registry/version.txt"
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            version = f.read().strip()
    else:
        version = "unknown"

    return species[prediction], version
