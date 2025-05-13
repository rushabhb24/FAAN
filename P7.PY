import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    return 1 if x >= 0 else 0

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=10):
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.lr = learning_rate
        self.epochs = epochs

    def train(self, X, y):
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}")
            for i in range(len(X)):
                x_i = X[i]
                y_pred = step_function(np.dot(x_i, self.weights) + self.bias)
                error = y[i] - y_pred
                self.weights += self.lr * error * x_i
                self.bias += self.lr * error
                print(f"Input: {x_i}, Predicted: {y_pred}, Actual: {y[i]}, Error: {error}")
            print(f"Weights: {self.weights}, Bias: {self.bias}\n")

    def predict(self, x):
        return step_function(np.dot(x, self.weights) + self.bias)

# Linearly separable data (e.g., Class 0 and Class 1)
X = np.array([
    [2, 1],
    [1, 3],
    [2, 3],
    [4, 2],
    [3, 6],
    [5, 4]
])
y = np.array([0, 0, 0, 1, 1, 1])

# Train perceptron
model = Perceptron(input_size=2, epochs=20)
model.train(X, y)

# Test predictions
print("Testing after training:")
for i in range(len(X)):
    print(f"Input: {X[i]}, Output: {model.predict(X[i])}")

# Plotting the data and decision boundary
def plot_decision_boundary(model, X, y):
    plt.figure()
    for i in range(len(X)):
        if y[i] == 0:
            plt.scatter(X[i][0], X[i][1], color='blue', label='Class 0' if i == 0 else "")
        else:
            plt.scatter(X[i][0], X[i][1], color='red', label='Class 1' if i == 3 else "")

    x_vals = np.linspace(0, 6, 100)
    y_vals = -(model.weights[0] * x_vals + model.bias) / model.weights[1]
    plt.plot(x_vals, y_vals, '--k', label='Decision Boundary')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Linearly Separable Classification")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_decision_boundary(model, X, y)
