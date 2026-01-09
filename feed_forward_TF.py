import tensorflow as tf

# ---- DATA ----
x = tf.constant([[0.1, 0.2, 0.3]], dtype=tf.float32)   
y = tf.constant([[0.7, 0.9]], dtype=tf.float32)        

# ---- VARIABLES (weights and bias) ----
W = tf.Variable([[0.3, 0.5],
                 [0.1, 0.4],
                 [0.7, 0.6]], dtype=tf.float32)
b = tf.Variable([0.4, 0.1], dtype=tf.float32)

lr = 0.01
epochs = 100

# ---- SIGMOID FUNCTIONS ----
def sigmoid(x):
    return 1 / (1 + tf.exp(-x))

def sigmoid_derivative(y_hat):
    return y_hat * (1 - y_hat)

# ---- TRAINING LOOP ----
for _ in range(epochs):
    with tf.GradientTape() as tape:
        z = tf.matmul(x, W) + b
        y_hat = sigmoid(z)
        loss = y - y_hat        

    # manual backward
    dz = -(y - y_hat) * sigmoid_derivative(y_hat)
    dW = tf.matmul(tf.transpose(x), dz)
    db = dz[0]

    # update weights
    W.assign_sub(lr * dW)
    b.assign_sub(lr * db)

# ---- FINAL OUTPUT AND SCALAR LOSS ----
z = tf.matmul(x, W) + b
y_hat = sigmoid(z)
final_loss = tf.reduce_mean(y - y_hat)  

print("Final output:", y_hat.numpy())
print("Final scalar loss:", final_loss.numpy())
