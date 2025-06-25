import mlflow
import mlflow.sklearn

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:8081")

# Name of the registered model
model_name = "RandomForest_Classifier"

# Correct: load version 18 of the registered model
model = mlflow.sklearn.load_model("models:/RandomForest_Classifier@best_model")

# Example input for prediction (for dataset version v3: all 4 features)
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)

print("✅ Prediction:", prediction)
