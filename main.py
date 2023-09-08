# İmport the "tensorflow", "numpy" and "matplotlib" libraries.
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Loading the "Fashion MNIST" dataset.
mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Scaling the images in the dataset.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Create the model.
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Compile the model.
model.compile(optimizer='rmsprop',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Fit the model.
model.fit(train_images, train_labels, epochs=20)

# Learning the accuracy of the model.
_, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nAccuracy:', test_acc*100,'%')

# Making predictions with the model.
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
prediction = probability_model.predict(test_images)
print(np.argmax(prediction[10]))









