from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
import shutil

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# standard train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# make dir to store models if not already there
os.makedirs("model_registry", exist_ok=True)

# use timestamp in filename so we know when it was trained
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
versioned_path = f"model_registry/iris_model_{timestamp}.pkl"
joblib.dump(model, versioned_path)

# overwrite latest_model.pkl so api always gets newest
latest_path = "model_registry/latest_model.pkl"
shutil.copy(versioned_path, latest_path)

# write the version info for API visibility
with open("model_registry/version.txt", "w") as f:
    f.write(f"iris_model_{timestamp}.pkl")
