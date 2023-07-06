import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MLP_classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        # ========================= EDIT HERE =========================
        # Define the layers of the model.
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, num_classes)
        self.relu = nn.Tanh()
        # Define the activation function.
        self.activation = nn.Softmax(dim=1)
        # =============================================================

    def forward(self, x):
        # ========================= EDIT HERE =========================
        # Given the input x, the function should return the output of the model.
        # Depending on the activation function, the output may vary.
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.activation(x)
        # =============================================================
        return x

    def train(self, train_loader, test_loader, criterion, optimizer, num_epochs):
        loss = None  # loss of final epoch
        # ========================= EDIT HERE =========================
        # Train should be done for 'num_epochs' times.
        # Define the loss function.
        # Define the optimizer.
        # Start training.
        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            # =============================================================
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
            self.evaluate(test_loader)

    def evaluate(self, test_loader):
        correct = 0
        total = 0
        predicted_labels = []
        # real_labels = []
        # ========================= EDIT HERE =========================
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                predicted_labels.append(predicted)
                # real_labels.extend(labels)
        predicted_labels = torch.cat(predicted_labels, 0)
        # =============================================================
        print(f'Accuracy: {100 * correct / total:.2f}%')
        # acc = sum(predicted_labels == torch.Tensor(real_labels)).item()/5000
        # print(f'Accuracy: {acc:.4f}%')
        return predicted_labels

    def predict(self, test_loader):
        correct = 0
        total = 0
        predicted_labels = []
        # real_labels = []
        # ========================= EDIT HERE =========================
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.forward(inputs)
                predicted_labels.append(outputs)
                # real_labels.extend(labels)
        predicted_labels = torch.cat(predicted_labels, 0)
        # =============================================================

        return predicted_labels

    def predict_proba(self, inputs):
        predicted_labels = []
        # real_labels = []
        # ========================= EDIT HERE =========================
        with torch.no_grad():
            predicted_labels = self.forward(inputs)
            # real_labels.extend(labels)
        # =============================================================
        return predicted_labels
