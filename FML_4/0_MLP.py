import numpy as np
import matplotlib.pyplot as plt
from models.functions import ReLU, Linear, Sigmoid, SigmoidCELayer
from utils import load_spiral

"""
** MLP Class is already implemented **
** Adjust Hyperparameters and experiment **
"""
np.random.seed(42)

class MLP:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size=1):
        
        # =============== EDIT HERE ===============
        

        self.activation1 = ReLU() # Sigmoid(), ReLU()
        self.activation2 = Sigmoid() # Sigmoid(), ReLU()

        # =============== EDIT HERE ===============


        self.input_layer = Linear(input_size, hidden_size1)
        self.hidden_layer = Linear(hidden_size1, hidden_size2)
        self.output_layer = SigmoidCELayer(hidden_size2, output_size)

    def predict(self, x):
        x = self.input_layer.forward(x)
        x = self.activation1.forward(x)
        x = self.hidden_layer.forward(x)
        x = self.activation2.forward(x)
        pred = self.output_layer.predict(x)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        return pred

    def loss(self, x, y):
        x = self.input_layer.forward(x)
        x = self.activation1.forward(x)
        x = self.hidden_layer.forward(x)
        x = self.activation2.forward(x)
        loss = self.output_layer.forward(x, y)
        return loss

    def gradient(self):
        d_prev = 1
        d_prev = self.output_layer.backward(d_prev=d_prev)
        d_prev = self.activation2.backward(d_prev=d_prev)
        d_prev = self.hidden_layer.backward(d_prev=d_prev)
        d_prev = self.activation1.backward(d_prev=d_prev)
        self.input_layer.backward(d_prev)
    def update(self, learning_rate, batch_size):
        self.input_layer.W -= self.input_layer.dW * learning_rate / batch_size
        self.input_layer.b -= self.input_layer.db * learning_rate / batch_size
        self.hidden_layer.W -= self.hidden_layer.dW * learning_rate / batch_size
        self.hidden_layer.b -= self.hidden_layer.db * learning_rate / batch_size
        self.output_layer.W -= self.output_layer.dW.sum(axis=-1).reshape(-1,1) * learning_rate
        self.output_layer.b -= self.output_layer.db.sum() * learning_rate


# =============== EDIT HERE ===============
hidden_dim1 = 100
hidden_dim2 = 100
num_epochs = 1000
print_every = 100
batch_size = 128
learning_rate = 0.03


"""
If you want to not use learning rate weight decay, you should define it as 1.
However, since the optimizer is not being used in the current code, I would recommend using a decay rate 
# recommended hyper-parameter : 0.98 ~ 1
"""
learning_rate_decay_rate = 0.99

# =========================================
train_acc = []
test_acc = []

x_train, y_train, x_test, y_test = load_spiral('./data')

num_feature = x_train.shape[1]
model = MLP(input_size=num_feature, hidden_size1=hidden_dim1, hidden_size2=hidden_dim2 , output_size=1)

num_data = len(x_train)
num_batch = int(np.ceil(num_data / batch_size))
for i in range(1, num_epochs + 1):
    epoch_loss = 0.0
    for b in range(0, len(x_train), batch_size):
        x_batch = x_train[b: b + batch_size]
        y_batch = y_train[b: b + batch_size]

        loss = model.loss(x_batch, y_batch)
        epoch_loss += loss

        model.gradient()
        model.update(learning_rate, batch_size)
    epoch_loss /= num_batch

    # Train accuracy
    pred = model.predict(x_train)
    pred = pred.reshape(-1)

    total = len(x_train)
    correct = len(np.where(pred == y_train)[0])
    tr_acc = correct / total
    train_acc.append(tr_acc)

    # Test accuracy
    pred = model.predict(x_test)
    pred = pred.reshape(-1)

    total = len(x_test)
    correct = len(np.where(pred == y_test)[0])
    te_acc = correct / total
    train_acc.append(te_acc)

    if i % print_every == 0:
        print('[EPOCH %d] Loss = %f' % (i, epoch_loss))
        print('Train Accuracy = %.3f' % tr_acc)
        print('Test Accuracy = %.3f' % te_acc)
    learning_rate *= learning_rate_decay_rate