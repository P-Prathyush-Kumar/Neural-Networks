import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

# Load MNIST dataset 

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data() 

# Normalize (0–255 → 0–1)
X_train = X_train / 255.0 
X_test = X_test / 255.0  

# Build model 

model = tf.keras.models.Sequential([ 
    tf.keras.layers.Flatten(input_shape=(28, 28)), 
    tf.keras.layers.Dense(128, activation='relu'), 
    tf.keras.layers.Dense(10, activation='softmax')  # digits 0–9 
])

model.compile( 
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy'] 
) 

# Train

model.fit(X_train, y_train, epochs=5)

# Test accuracy

model.evaluate(X_test, y_test)

# SINGLE DIGIT PREDICTION

index = 0   # change index to test another digit

digit = X_test[index]
prediction = model.predict(digit.reshape(1, 28, 28))
predicted_digit = np.argmax(prediction)

plt.imshow(digit, cmap='gray')
plt.title(f"Predicted Digit: {predicted_digit}")
plt.axis('off')
plt.show()
