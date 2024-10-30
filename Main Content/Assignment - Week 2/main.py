# Import necessary libraries
import time
start_time = time.time()

import os
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
from torch.utils.data import DataLoader, TensorDataset

os.system('cls' if os.name == 'nt' else 'clear')

# Parameters
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

# Load and preprocess the Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# CNN Model (PyTorch) - Code 1
class CNNModel:
    def __init__(self):
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Conv2d(32, 64, 3), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Flatten(),
            nn.Linear(64 * 6 * 6, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def train(self, X_train, y_train):
        X_train_tensor = torch.tensor(X_train).unsqueeze(1)
        y_train_tensor = torch.tensor(y_train)
        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True)
        
        self.model.train()
        for epoch in range(EPOCHS):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(batch_X), batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {total_loss / len(train_loader):.4f}")

    def evaluate(self, X, y):
        self.model.eval()
        with torch.no_grad():
            X_tensor, y_tensor = torch.tensor(X).unsqueeze(1), torch.tensor(y)
            accuracy = (torch.argmax(self.model(X_tensor), dim=1) == y_tensor).float().mean().item()
        return accuracy

# Preprocess data for both models
X_train_flat = X_train.reshape(-1, 28, 28)
X_test_flat = X_test.reshape(-1, 28, 28)

# Instantiate and train both models
print("Training Intiated...\n")
cnn_model = CNNModel()
cnn_model.train(X_train_flat, y_train)
cnn_accuracy = cnn_model.evaluate(X_test_flat, y_test)

# Results
print(f"\nCNN Model Test Accuracy: {cnn_accuracy * 100:.2f}%")

end_time = time.time()

print(f"\nTotal Runtime: {end_time - start_time :.4f} seconds")