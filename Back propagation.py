import numpy as np
# NumPy library

arr1 = np.array([1, 2, 3])
# Input vector

W = np.array([[1, 2],
              [3, 4],
              [5, 6]])
# Weight matrix

b = np.array([8, 4])
# Bias vector

z = np.dot(arr1, W) + b
# Weighted sum

print("z:", z)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# Sigmoid activation

print("sigmoid:", sigmoid(z))

y = np.array([1, 0])
# Target output

loss = -(y * np.log(sigmoid(z)) + (1 - y) * np.log(1 - sigmoid(z)))
# Loss function

print("loss:", loss)

dz = sigmoid(z) - y
# Output gradient

dW = np.outer(arr1, dz)
# Weight gradient

db = dz
# Bias gradient

print("dz:", dz)
print("dW:\n", dW)
print("db:", db)

lr = 0.01
# Learning rate

W = W - lr * dW
# Update weights

b = b - lr * db
# Update bias

print("Updated W:\n", W)
print("Updated b:", b)
