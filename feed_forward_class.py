import numpy as np

class NeuralNet:
    def __init__(self, W, b, lr=0.01):
        self.W = W
        self.b = b
        self.lr = lr

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, y):
        return y * (1 - y)

    def forward(self, x):
        self.x = x
        self.z = np.dot(x, self.W) + self.b
        self.y_hat = self.sigmoid(self.z)
        return self.y_hat

    def compute_loss(self, y):
        self.y = y
        loss = y - self.y_hat
        return loss

    def backward(self):
        dz = -(self.y - self.y_hat) * self.sigmoid_derivative(self.y_hat)
        dW = np.outer(self.x, dz)
        db = dz

        self.W -= self.lr * dW
        self.b -= self.lr * db

    def train(self, x, y, epochs=100):
        for _ in range(epochs):
            self.forward(x)
            self.compute_loss(y)
            self.backward()


# usage
x = np.array(range(1,4))
W = np.array([[1,2],[3,4],[5,6]], dtype=float)
b = np.array([8,4], dtype=float)
y = np.array([1,0])

nn = NeuralNet(W, b, lr=0.01)
nn.train(x, y)

print(nn.forward(x))
