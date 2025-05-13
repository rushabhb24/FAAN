import numpy as np

# Define the step activation function
def step_function(x):
    return np.where(x >= 0, 1, 0)

# Perceptron training function
def train_perceptron(X, y, lr=0.1, epochs=10):
    weights = np.zeros(X.shape[1])
    bias = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        for i in range(len(X)):
            linear_output = np.dot(X[i], weights) + bias
            y_pred = step_function(linear_output)
            error = y[i] - y_pred

            # Update rule
            weights += lr * error * X[i]
            bias += lr * error

            print(f"Sample: {X[i]}, Label: {y[i]}, Predicted: {y_pred}, Error: {error}")
        print(f"Weights: {weights}, Bias: {bias}\n")

    return weights, bias

# Prediction function
def predict(X, weights, bias):
    return step_function(np.dot(X, weights) + bias)

# Sample Input (linearly separable)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 0, 0, 1])  # AND gate example

# Training
weights, bias = train_perceptron(X, y)

# Testing
for i in range(len(X)):
    print(f"Input: {X[i]}, Prediction: {predict(X[i], weights, bias)}")
