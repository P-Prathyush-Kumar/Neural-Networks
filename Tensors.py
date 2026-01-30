import tensorflow as tf

a = tf.constant([[1, 2],
                 [3, 4]])

b = tf.constant([[5, 6],
                 [7, 8]])


print(tf.reshape(a, (4,)))  # reshape
print(tf.reduce_sum(a))     # sum
print(tf.matmul(a, a))      # matrix multiplication

# basic info
print(a.shape)
print(b.shape)

# operations on two tensors
print(tf.add(a, b))             # sum
print(tf.subtract(a, b))        # difference
print(tf.multiply(a, b))        # element-wise
print(tf.reduce_mean(a + b))    # mean
print(tf.matmul(a, b))          # matrix multiplication
