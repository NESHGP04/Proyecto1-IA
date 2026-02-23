import numpy as np

class MLP:
    def __init__(self, input_size, hidden1, hidden2, output_size):
        self.W1 = np.random.randn(input_size, hidden1) * np.sqrt(2 / input_size)
        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2 / hidden1)
        self.W3 = np.random.randn(hidden2, output_size) * np.sqrt(2 / hidden2)
        self.b1 = np.zeros((1, hidden1))

        self.W2 = np.random.randn(hidden1, hidden2) * 0.01
        self.b2 = np.zeros((1, hidden2))

        self.W3 = np.random.randn(hidden2, output_size) * 0.01
        self.b3 = np.zeros((1, output_size))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_deriv(self, Z):
        return Z > 0

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.relu(self.Z2)

        self.Z3 = self.A2 @ self.W3 + self.b3
        self.A3 = self.softmax(self.Z3)

        return self.A3

    def backward(self, X, y, lr):
        m = X.shape[0]

        dZ3 = self.A3 - y
        dW3 = self.A2.T @ dZ3 / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * self.relu_deriv(self.Z2)
        dW2 = self.A1.T @ dZ2 / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.relu_deriv(self.Z1)
        dW1 = X.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W3 -= lr * dW3
        self.b3 -= lr * db3
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def train(self, X, y, epochs=3000, lr=0.01):
        for epoch in range(epochs):
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            self.forward(X_shuffled)
            self.backward(X_shuffled, y_shuffled, lr)

            if epoch % 500 == 0:
                preds = self.predict(X)
                true = np.argmax(y, axis=1)
                acc = np.mean(preds == true)
                print(f"Epoch {epoch} - Train Accuracy: {acc}")

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)