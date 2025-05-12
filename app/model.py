import joblib
import pandas as pd
import os

# Load model
model_path = "model_registry/latest_model.pkl"
model = joblib.load(model_path)

# Load mock feature store
feature_store = pd.read_csv("feature_store/feature_store.csv")

def get_features_by_id(feature_id: int):
    row = feature_store[feature_store['feature_id'] == feature_id]
    if row.empty:
        raise ValueError(f"Feature ID {feature_id} not found in feature store.")
    return row[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values.flatten().tolist()

def predict_species(features: list) -> tuple:
    prediction = model.predict([features])[0]
    species = ["setosa", "versicolor", "virginica"]
    return species[prediction], os.path.basename(model_path)
