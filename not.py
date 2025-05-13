import numpy as np

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
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = step_function(linear_output)
                error = y[i] - y_pred

                # Update rule
                self.weights += self.lr * error * x_i
                self.bias += self.lr * error

                print(f"Input: {x_i}, Predicted: {y_pred}, Target: {y[i]}, Error: {error}")
            print(f"Weights: {self.weights}, Bias: {self.bias}\n")

    def predict(self, x):
        return step_function(np.dot(x, self.weights) + self.bias)

# NOT gate input and output
X = np.array([[0], [1]])
y = np.array([1, 0])  # Output of NOT gate

# Train perceptron
model = Perceptron(input_size=1)
model.train(X, y)

# Test predictions
print("Testing after training:")
for i in range(len(X)):
    print(f"Input: {X[i]}, Output: {model.predict(X[i])}")
