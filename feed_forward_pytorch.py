import torch
import torch.nn as nn

# ---- DATA ----
x = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32) 
y = torch.tensor([[0.7, 0.9]], dtype=torch.float32)       

# ---- VARIABLES (weights and bias) ----
W = torch.tensor([[0.3, 0.5],
                  [0.1, 0.4],
                  [0.7, 0.6]], dtype=torch.float32, requires_grad=True)
b = torch.tensor([0.4, 0.1], dtype=torch.float32, requires_grad=True)

lr = 0.01
epochs = 100

# ---- SIGMOID FUNCTIONS ----
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def sigmoid_derivative(y_hat):
    return y_hat * (1 - y_hat)

# ---- TRAINING LOOP ----
for _ in range(epochs):
    # forward
    z = torch.matmul(x, W) + b
    y_hat = sigmoid(z)
    loss = y - y_hat  

    # backward (manual)
    dz = -(y - y_hat) * sigmoid_derivative(y_hat)
    dW = torch.matmul(x.T, dz)
    db = dz[0]

    # update weights
    W.data -= lr * dW
    b.data -= lr * db

# ---- FINAL OUTPUT AND SCALAR LOSS ----
z = torch.matmul(x, W) + b
y_hat = sigmoid(z)
final_loss = torch.mean(y - y_hat)  

print("Final output:", y_hat.detach().numpy())
print("Final scalar loss:", final_loss.item())
