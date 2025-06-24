import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Set MLflow Tracking URI (optional if using local UI)
mlflow.set_tracking_uri("http://127.0.0.1:8080")  # or leave it out if not needed

# Set MLflow experiment
mlflow.set_experiment("Mushroom Classification")

# Load dataset
data = pd.read_csv("D:/OneDrive/Desktop/Model-version/class-1/mushrooms.csv")

# Encode features
X = pd.get_dummies(data.drop("class", axis=1))

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["class"])

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model parameters
params = {
    "penalty": "l2",
    "solver": "newton-cg",
    "multi_class": "auto",
    "max_iter": 200
}
mlflow.set_experiment("Get started")
# Start MLflow run
with mlflow.start_run():
    # Log model parameters
    mlflow.log_params(params)
    
    # Train model
    lr = LogisticRegression(**params)
    lr.fit(x_train, y_train)

    # Predict and evaluate
    y_pred = lr.predict(x_test)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Print results
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    # Log the model
    mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="model",
        input_example=x_train,
        registered_model_name="getting_started_model"
    )
