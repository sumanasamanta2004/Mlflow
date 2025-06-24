import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load and preprocess data
data = pd.read_csv("D:/OneDrive/Desktop/Model-version/class-1/mushrooms.csv")
X = pd.get_dummies(data.drop("class", axis=1)).values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["class"])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=32, shuffle=True)

# Define PyTorch model
class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = Net(input_dim=x_train.shape[1], output_dim=len(set(y)))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("Mushroom PyTorch Classification")

with mlflow.start_run():
    # Training loop
    model.train()
    for epoch in range(10):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        y_pred_logits = model(x_test_tensor)
        y_pred_labels = y_pred_logits.argmax(dim=1).numpy()

    acc = accuracy_score(y_test, y_pred_labels)
    precision = precision_score(y_test, y_pred_labels, average='macro')
    recall = recall_score(y_test, y_pred_labels, average='macro')
    f1 = f1_score(y_test, y_pred_labels, average='macro')

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Log model
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        input_example=x_test_tensor[:1].numpy(),
        registered_model_name="getting_started_pytorch_model"
    )
