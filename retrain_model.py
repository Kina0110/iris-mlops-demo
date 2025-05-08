# retrain_model.py

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "app/iris_model.pkl")

print("Model retrained and saved to app/iris_model.pkl")


# The model was trained using scikit-learn 1.1.3.
# Your environment is running scikit-learn 1.5.1.
# to match envireents run this script ( if issue occurs)