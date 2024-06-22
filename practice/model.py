import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_data(X: torch.Tensor):
    plt.scatter([i for i in range(len(X))], X, c="b", s=4)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()

weight = 0.7 # represent b in the previous formula
bias = 0.3 # represent a in the previous formula

# build a model that estimates those numbers
start = 0
end = 9
step = 0.05
X = torch.cos(torch.arange(start, end, step).unsqueeze(dim=1))
Y = weight * X + bias

# train/test data
train_split = int(0.8 * len(X)) # 40

X_train, Y_train = X[:train_split], Y[:train_split]
X_test, Y_test = X[train_split:], Y[train_split:]

def plot_predictions(train_data=X_train, train_labels = Y_train, test_data=X_test, test_labels=Y_test, predictions=None):
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data") # c: color s: size label: legend

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})
    plt.show()

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

torch.manual_seed(42)
model = LinearRegressionModel()

with torch.inference_mode():
    y_preds = model(X_test)
    
# plot_predictions(predictions=y_preds)

# setup a loss function (Mean absolute error)
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

epochs = 500

# Training
for epoch in range(epochs):
    # set the model to training mode
    model.train()

    # 1. forward pass
    y_pred = model(X_train)

    loss = loss_fn(y_pred, Y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model.eval()

plot_predictions(predictions=y_preds)