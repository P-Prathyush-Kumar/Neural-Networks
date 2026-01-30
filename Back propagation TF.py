import tensorflow as tf

# Input and target
X = tf.constant([[1, 2, 3]], dtype=tf.float32)
y = tf.constant([[1, 0]], dtype=tf.float32)

# Smaller weights and bias
W = tf.Variable([[0.1, 0.2],
                 [0.3, 0.4],
                 [0.5, 0.6]], dtype=tf.float32)
b = tf.Variable([0.1, 0.2], dtype=tf.float32)

# Forward pass and stable loss
with tf.GradientTape() as tape:
    z = tf.matmul(X, W) + b
    output = tf.sigmoid(z)
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=z))

# Gradients
dW, db = tape.gradient(loss, [W, b])

# Update
lr = 0.01
W.assign_sub(lr * dW)
b.assign_sub(lr * db)

# Output
print("z:", z.numpy())
print("sigmoid:", output.numpy())
print("loss:", loss.numpy())
print("dW:\n", dW.numpy())
print("db:", db.numpy())
print("Updated W:\n", W.numpy())
print("Updated b:", b.numpy())
