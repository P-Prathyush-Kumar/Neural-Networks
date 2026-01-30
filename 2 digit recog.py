import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset
(train_imgs, train_lbls), (test_imgs, test_lbls) = mnist.load_data()

# Normalize the images
train_imgs = train_imgs / 255.0
test_imgs = test_imgs / 255.0

# Build the Feed Forward Neural Network
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(150, activation='relu'),  # changed number of neurons
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(train_imgs, train_lbls, epochs=6)  # changed epochs

# Evaluate the model
loss, accuracy = model.evaluate(test_imgs, test_lbls)
print(f"Test Accuracy: {accuracy:.4f}")

# Select two test images
idx1, idx2 = 2, 3  # different indices
img1, lbl1 = test_imgs[idx1], test_lbls[idx1]
img2, lbl2 = test_imgs[idx2], test_lbls[idx2]

# Predict each digit
pred1 = model.predict(img1.reshape(1, 28, 28), verbose=0)
pred2 = model.predict(img2.reshape(1, 28, 28), verbose=0)

pred_lbl1 = tf.argmax(pred1[0]).numpy()
pred_lbl2 = tf.argmax(pred2[0]).numpy()

# Combine images horizontally
combined_img = np.hstack((img1, img2))

# Display the result
plt.figure(figsize=(8, 4))
plt.imshow(combined_img, cmap='gray')
plt.title(f"True: {lbl1}{lbl2}, Predicted: {pred_lbl1}{pred_lbl2}")
plt.axis('off')
plt.show()
