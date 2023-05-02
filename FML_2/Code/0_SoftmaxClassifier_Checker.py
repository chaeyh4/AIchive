import numpy as np
from optim.Optimizer import SGD
from models.SoftmaxClassifier import SoftmaxClassifier
from utils import accuracy


np.random.seed(428)

num_epochs = 50
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
y = np.array([2, 2, 2, 1, 1, 1, 0, 0])

num_data, num_features = x.shape
num_label = int(y.max()) + 1

print('# of Training data : %d \n' % num_data)

model = SoftmaxClassifier(num_features, num_label)

# ================================== Softmax ==================================
print('='*20, 'Softmax Function Test', '='*20)

"""
You should get results as:

Softmax out:
[[2.06106005e-09 4.53978686e-05 9.99954600e-01]
 [3.33333333e-01 3.33333333e-01 3.33333333e-01]
 [9.00305732e-02 2.44728471e-01 6.65240956e-01]
 [1.94615163e-03 3.90895004e-02 9.58964348e-01]]
"""

softmax_in = np.array([
    [-10, 0, 10],
    [0, 0, 0],
    [100, 101, 102],
    [-0.7, 2.3, 5.5]])

softmax_out = model._softmax(softmax_in)

print('Softmax out:')
print(softmax_out)
print()

# ==================================  Train  ==================================
print('='*20, 'SoftmaxClassifier Test', '='*20)

"""
Depending on your implementation, there may be differences from the results below. We will check your implementation manually.

Initial weight:
[[0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]]

Trained weight:
[[-49.13178884  16.4794766   32.65231224]
 [ 18.72917943 -23.77973772   5.05055829]
 [ 12.22912875  -1.66819996 -10.56092879]
 [ -5.14853415  19.08129052 -13.93275637]]

Accuracy on train data : 0.625000
"""

print('Initial weight:')
print(model.W)
print()

optim = SGD()
model.train(x, y, num_epochs, batch_size, learning_rate, optim)

print('Trained weight:')
print(model.W)
print()

pred, prob = model.eval(x)

train_acc = accuracy(pred, y)
print('Accuracy on train data : %f\n' % train_acc)
