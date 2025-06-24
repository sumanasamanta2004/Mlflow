import mlflow
import mlflow.keras
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature

# Load and preprocess data
data = pd.read_csv("D:/OneDrive/Desktop/Model-version/class-1/mushrooms.csv")
X = pd.get_dummies(data.drop("class", axis=1)).values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["class"])

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert labels to categorical
y_train_cat = tf.keras.utils.to_categorical(y_train)
y_test_cat = tf.keras.utils.to_categorical(y_test)

# Build TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(x_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(y_train_cat.shape[1], activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Setup MLflow
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("Mushroom TensorFlow Classification")

with mlflow.start_run():
    # Train model
    model.fit(x_train, y_train_cat, epochs=10, batch_size=32, verbose=0)

    # Predict
    y_pred_prob = model.predict(x_test)
    y_pred_labels = np.argmax(y_pred_prob, axis=1)

    # Metrics
    acc = accuracy_score(y_test, y_pred_labels)
    precision = precision_score(y_test, y_pred_labels, average='macro')
    recall = recall_score(y_test, y_pred_labels, average='macro')
    f1 = f1_score(y_test, y_pred_labels, average='macro')

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Prepare input example and signature
    input_example = x_test[:1]
    signature = infer_signature(x_test, y_pred_prob)

    # Log the model safely
    mlflow.keras.log_model(
        model=model,
        artifact_path="model",
        registered_model_name="getting_started_tf_model"
    )
