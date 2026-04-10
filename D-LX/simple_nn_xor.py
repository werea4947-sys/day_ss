import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(y):
    return y * (1 - y)


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, lr=0.5):
        self.lr = lr
        rng = np.random.default_rng(1)
        self.W1 = rng.normal(scale=1.0, size=(input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = rng.normal(scale=1.0, size=(hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def compute_loss(self, y_pred, y_true):
        return 0.5 * np.mean((y_pred - y_true) ** 2)

    def backward(self, X, y_true):
        m = X.shape[0]
        delta2 = (self.a2 - y_true) * sigmoid_deriv(self.a2)
        dW2 = self.a1.T @ delta2 / m
        db2 = np.mean(delta2, axis=0, keepdims=True)

        delta1 = (delta2 @ self.W2.T) * sigmoid_deriv(self.a1)
        dW1 = X.T @ delta1 / m
        db1 = np.mean(delta1, axis=0, keepdims=True)

        # update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=10000, print_every=1000):
        for e in range(1, epochs + 1):
            y_pred = self.forward(X)
            loss = self.compute_loss(y_pred, y)
            self.backward(X, y)
            if e % print_every == 0 or e == 1:
                print(f"Epoch {e:5d}: loss={loss:.6f}")


def demo_xor():
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    net = TwoLayerNet(input_size=2, hidden_size=4, output_size=1, lr=1.0)
    net.train(X, y, epochs=5000, print_every=500)

    preds = net.forward(X)
    print("\nFinal predictions:")
    for inp, p in zip(X, preds):
        print(f"{inp} -> {p[0]:.4f} (rounded {int(p[0]>0.5)})")


if __name__ == '__main__':
    demo_xor()
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(y):
    return y * (1 - y)


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, lr=0.5):
        self.lr = lr
        rng = np.random.default_rng(1)
        self.W1 = rng.normal(scale=1.0, size=(input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = rng.normal(scale=1.0, size=(hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def compute_loss(self, y_pred, y_true):
        return 0.5 * np.mean((y_pred - y_true) ** 2)

    def backward(self, X, y_true):
        m = X.shape[0]
        delta2 = (self.a2 - y_true) * sigmoid_deriv(self.a2)
        dW2 = self.a1.T @ delta2 / m
        db2 = np.mean(delta2, axis=0, keepdims=True)

        delta1 = (delta2 @ self.W2.T) * sigmoid_deriv(self.a1)
        dW1 = X.T @ delta1 / m
        db1 = np.mean(delta1, axis=0, keepdims=True)

        # update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=10000, print_every=1000):
        for e in range(1, epochs + 1):
            y_pred = self.forward(X)
            loss = self.compute_loss(y_pred, y)
            self.backward(X, y)
            if e % print_every == 0 or e == 1:
                print(f"Epoch {e:5d}: loss={loss:.6f}")


def demo_xor():
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    net = TwoLayerNet(input_size=2, hidden_size=4, output_size=1, lr=1.0)
    net.train(X, y, epochs=5000, print_every=500)

    preds = net.forward(X)
    print("\nFinal predictions:")
    for inp, p in zip(X, preds):
        print(f"{inp} -> {p[0]:.4f} (rounded {int(p[0]>0.5)})")


if __name__ == '__main__':
    demo_xor()
