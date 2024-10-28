# Build a CNN for the Fashion MNIST Dataset and Benchmark it against the previous Neural Network which used the same dataset (Week 1 Assignment)

from functions import * 
import tensorflow as tf

# Importing Fashion MNIST Dataset
(train_image_data, train_image_label), (test_image_data, test_image_label) = tf.keras.datasets.fashion_mnist.load_data()

# Checking the dataset shape
print("\nTraining Image Data: ", train_image_data.shape)
print("Training Image Labels: ", train_image_label.shape)
print("\nTesting Image Data: ", test_image_data.shape)
print("Testing Image Labels: ", test_image_label.shape)

# Applying Sobel Filter to the dataset
gradient_x, gradient_y, gradient_magnitude = apply_sobel_filter(train_image_data)

# Displaying the Results after applying the filters
display_results(train_image_data[0], gradient_x[0], gradient_y[0], gradient_magnitude[0])