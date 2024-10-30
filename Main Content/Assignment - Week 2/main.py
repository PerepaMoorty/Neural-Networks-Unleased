# Build a Ctorch.nn for the Fashion MNIST Dataset and Benchmark it against the previous Neural Network which used the same dataset (Week 1 Assignment)

import time
import os
import torch
import torch.nn
import torch.nn.functional
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Clearing Console
os.system("clear")

# Start Time
start_time = time.time()

# Importing Fashion MNIST Dataset
(train_image_data, train_image_label), (test_image_data, test_image_label) = tf.keras.datasets.fashion_mnist.load_data()
train_image_data = train_image_data.astype('float32') / 255.0
test_image_data = test_image_data.astype('float32') / 255.0

# Checking the dataset shape
print("Training Image Data: ", train_image_data.shape)
print("Training Image Labels: ", train_image_label.shape)
print("Testing Image Data: ", test_image_data.shape)
print("Testing Image Labels: ", test_image_label.shape)

# Defining the classes and the names
CLASS_NAMES = ["T-Shirt/Top", "Trousers", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

# Checking if GPU can be used
if(torch.cuda.is_available()):
    print("CUDA available, using GPU for Calculations")
    device = "cuda"
else:
    print("CUDA not available, using CPU for Calculations")
    device = "cpu"

# Hyper Parameters
EPOCHS = 10
LEARNING_RATE = 0.001
        
# Defining the CNN Model
class ConvolutionalNeuralNetwork_Model:
    def __init__(self):
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d((4, 4)),
            torch.nn.Conv2d(32, 16, kernel_size=3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2))
        )
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters, lr=LEARNING_RATE)
        
    def train(self, train_data, train_label):
        train_data_tensor = torch.tensor(train_data).unsqueeze(1)
        train_label_tensor = torch.tensor(train_label)

# End Time
end_time = time.time()

# Calculating the Total Runtime
print(f"\n\nTotal Runtime: {end_time - start_time : .4f}")