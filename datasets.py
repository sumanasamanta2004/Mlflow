import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Optional: Set MLflow tracking URI (uncomment if using a custom server)
mlflow.set_tracking_uri("http://127.0.0.1:8081")

# Set MLflow experiment
mlflow.set_experiment("Iris_Model_Experiment_Tracking")

# Load the IRIS dataset from CSV
df = pd.read_csv("iris.csv")

# Optional: rename columns to standard format (if needed)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Define dataset versions
versions = {
    "v1": df[['sepal_length','sepal_width', 'species']],
    "v2": df[['petal_length', 'petal_width', 'species']],
    "v3": df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']]
}

# Define hyperparameter sets for each model
hyperparams = {
    "LogisticRegression": [
        {"penalty": "l2", "solver": "lbfgs", "multi_class": "multinomial", "max_iter": 200},
        {"penalty": "none", "solver": "saga", "multi_class": "ovr", "max_iter": 300},
        {"penalty": "l1", "solver": "liblinear", "multi_class": "ovr", "max_iter": 150}
    ],
    "DecisionTree": [
        {"criterion": "gini", "max_depth": 3},
        {"criterion": "entropy", "max_depth": 5},
        {"criterion": "gini", "max_depth": None}
    ],
    "KNN": [
        {"n_neighbors": 3, "weights": "uniform"},
        {"n_neighbors": 5, "weights": "distance"},
        {"n_neighbors": 7, "weights": "uniform"}
    ],
    "RandomForest": [
        {"n_estimators": 50, "max_depth": 3},
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 150, "max_depth": None}
    ]
}

# Model class mapping
model_classes = {
    "LogisticRegression": LogisticRegression,
    "DecisionTree": DecisionTreeClassifier,
    "KNN": KNeighborsClassifier,
    "RandomForest": RandomForestClassifier
}

# Loop through each dataset version
for version_name, data in versions.items():
    X = data.drop('species', axis=1)
    y = data['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Loop through each model type and its hyperparameter sets
    for model_name, ModelClass in model_classes.items():
        for i, params in enumerate(hyperparams[model_name]):
            run_name = f"{model_name}_{version_name}_Set{i+1}"

            with mlflow.start_run(run_name=run_name):
                try:
                    # Train model
                    model = ModelClass(**params)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Calculate evaluation metrics
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
                    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

                    # Log parameters, metrics, tags
                    mlflow.set_tag("model", model_name)
                    mlflow.set_tag("dataset_version", version_name)
                    mlflow.set_tag("hyperparam_set", f"Set{i+1}")
                    mlflow.log_params(params)
                    mlflow.log_metrics({
                        "accuracy": acc,
                        "precision": prec,
                        "recall": rec,
                        "f1_score": f1
                    })

                    # Log the trained model
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path="model",
                        input_example=X_train.iloc[:5],
                        registered_model_name=f"{model_name}_Classifier"
                    )

                    # Optional: log a description file
                    with open("run_description.txt", "w") as f:
                        f.write(f"{run_name}: trained on {version_name} with {model_name}")
                    mlflow.log_artifact("run_description.txt")

                    print(f"✅ Logged: {run_name} | Accuracy: {acc:.4f}")

                except Exception as e:
                    print(f"❌ Error in {run_name}: {e}")
