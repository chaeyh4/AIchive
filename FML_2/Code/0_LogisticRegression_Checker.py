import numpy as np
from optim.Optimizer import SGD
from models.LogisticRegression import LogisticRegression
from utils import accuracy, rel_error


np.random.seed(428)

num_epochs = 30
batch_size = 3
learning_rate = 2

x = np.array([[1, 2, 1, 1],
            [2, 1, 3, 2], 
            [3, 1, 3, 4], 
            [4, 1, 5, 5], 
            [1, 7, 5, 5], 
            [1, 2, 5, 6], 
            [1, 6, 6, 6], 
            [1, 7, 7, 7]])
y = np.array([1, 1, 0, 1, 1, 1, 0, 0])

num_data, num_features = x.shape
num_label = int(y.max()) + 1

print('# of Training data : %d \n' % num_data)

model = LogisticRegression(num_features)

# ================================== Sigmoid ==================================
print('='*20, 'Sigmoid Function Test', '='*20)

"""
You should get results as:

Sigmoid out:
[[4.53978687e-05 5.00000000e-01 9.99954602e-01]
 [5.00000000e-01 5.00000000e-01 5.00000000e-01]
 [1.00000000e+00 1.00000000e+00 1.00000000e+00]
 [3.31812228e-01 9.08877039e-01 9.95929862e-01]]
 
 array([[4.53978687e-05, 5.00000000e-01, 9.99954602e-01],
       [5.00000000e-01, 5.00000000e-01, 5.00000000e-01],
       [1.00000000e+00, 1.00000000e+00, 1.00000000e+00],
       [3.31812228e-01, 9.08877039e-01, 9.95929862e-01]])
"""

sigmoid_in = np.array([
    [-10, 0, 10],
    [0, 0, 0],
    [100, 101, 102],
    [-0.7, 2.3, 5.5]])

sigmoid_out = model._sigmoid(sigmoid_in)
print('Sigmoid out:')
print(sigmoid_out)
print()

# ==================================  Train  ==================================
print('='*20, 'LogsticRegression Test', '='*20)

"""
Depending on your implementation, there may be differences from the results below. We will check your implementation manually.

Initial weight:
[[0.]
 [0.]
 [0.]
 [0.]]

[[ 21.99147772]
 [-19.54966854]
 [ -4.06577204]
 [ -7.97751466]]

Accuracy on train data : 0.375000
"""

print('Initial weight:')
print(model.W)
print()

optim = SGD()
model.train(x, y, batch_size, num_epochs, learning_rate, optim)

print('Trained weight:')
print(model.W)
print()

pred, prob = model.eval(x)

train_acc = accuracy(pred, y)
print('Accuracy on train data : %f\n' % train_acc)
