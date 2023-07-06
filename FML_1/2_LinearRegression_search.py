import numpy as np
from models.LinearRegression import LinearRegression

from utils import RMSE, load_data, optimizer


np.random.seed(2023)

"""
Choose param to search. (batch_size or lr or gamma for Momentum)
Specify values of the parameter to search,
and fix the other.
e.g.)
search_param = 'lr'
_batch_size = 32
_lr = [0.1, 0.01, 0.05]
"""

# You can EDIT the hyperparameters below.
_epoch = 200
search_param = 'lr'
_batch_size = 31
_lr = [0.0051]
_optim = 'Momentum'                    # Write one of SGD, or Momentum
_gamma = 0.96
_normalize = 'ZScore'                       # Write one of ZScore, MinMax, or None(Default).

_dataset_name = 'CCPP'                # Write one of Airbnb, RealEstate, or CCPP




# Data generation
train_data, test_data = load_data(_dataset_name, _normalize)
x_train_data, y_train_data = train_data[0], train_data[1]
x_test_data, y_test_data = test_data[0], test_data[1]

train_results = []
test_results = []
if search_param == 'lr':
    search_space = _lr
elif search_param == 'batch_size':
    search_space = _batch_size
elif search_param == 'gamma':
    search_space = _gamma
else:
    pass


total_errors = []
for i, space in enumerate(search_space):
    # Build model
    model = LinearRegression(num_features=x_train_data.shape[1])
    optim = optimizer(_optim, _gamma)

    # Train model with gradient descent
    if search_param == 'lr':
        model.numerical_solution(x=x_train_data, y=y_train_data, epochs=_epoch, batch_size=_batch_size, lr=space, optim=optim)
    elif search_param == 'batch_size':
        model.numerical_solution(x=x_train_data, y=y_train_data, epochs=_epoch, batch_size=space, lr=_lr, optim=optim)
    elif search_param == 'gamma':
        optim = optimizer(_optim, space)
        model.numerical_solution(x=x_train_data, y=y_train_data, epochs=_epoch, batch_size=_batch_size, lr=_lr, optim=optim)

    
    ################### Evaluate on train data
    # Inference
    inference = model.eval(x_train_data)

    # Assess model
    error = RMSE(inference, y_train_data)
    print('[Search %d] RMSE on Train Data : %.4f' % (i+1, error))

    ################### Evaluate on test data
    # Inference
    inference = model.eval(x_test_data)

    # Assess model
    error = RMSE(inference, y_test_data)
    print('[Search %d] RMSE on Test Data : %.4f' % (i+1, error))

    test_results.append(error)
