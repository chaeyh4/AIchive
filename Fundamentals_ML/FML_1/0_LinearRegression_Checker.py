import numpy as np
from models.LinearRegression import LinearRegression
from utils import optimizer, RMSE


np.random.seed(10)

Dataset = np.loadtxt('data/check_data.txt')
x_data, y_data = Dataset[:, :-1], Dataset[:, -1]

_epoch = 100
_batch_size = 5
_lr = 0.01
_optim = 'SGD'
_gamma = 0.2
#======================================================================================================

print('='*20, 'Batch Gradient Numerical Solution Test', '='*20)

Numeric_model = LinearRegression(num_features=x_data.shape[1])
optim = optimizer(_optim, _gamma)
print('Initial weight: \n', Numeric_model.W.reshape(-1))
print()

Numeric_model.numerical_solution(x=x_data, y=y_data, epochs=_epoch, batch_size=_batch_size, lr=_lr, optim=optim, batch_gradient=True)
print('Trained weight: \n', Numeric_model.W)
print()

# model evaluation
inference = Numeric_model.eval(x_data)

# Error calculation
error = RMSE(inference, y_data)
print('RMSE on Train Data : %.4f \n' % error)



"""
You should get results as:

Initial weight:
 [0. 0. 0. 0.]

Trained weight:
 [ 5.27425666 -3.63611817  4.0010919   6.55201443]

RMSE on Train Data : 56.6210
"""


#======================================================================================================

#======================================================================================================

print('='*20, 'Mini-Batch Stochastic Gradient Numerical Solution Test', '='*20)

Numeric_model = LinearRegression(num_features=x_data.shape[1])
print('Initial weight: \n', Numeric_model.W.reshape(-1))
print()

Numeric_model.numerical_solution(x=x_data, y=y_data, epochs=_epoch, batch_size=_batch_size, lr=_lr, optim=optim, batch_gradient=False)
print('Trained weight: \n', Numeric_model.W)
print()

# model evaluation
inference = Numeric_model.eval(x_data)

# Error calculation
error = RMSE(inference, y_data)
print('RMSE on Train Data : %.4f \n' % error)

"""
You should get results as:

Initial weight:
 [0. 0. 0. 0.]

Trained weight:
 [ 8.88790101 -6.56829237  8.46605373 11.63923295]

RMSE on Train Data : 48.6090
"""

#======================================================================================================

print('='*20, 'Analytic Solution Test', '='*20)

Analytic_model = LinearRegression(num_features=x_data.shape[1])
print('Initial weight: \n', Analytic_model.W.reshape(-1))
print()

Analytic_model.analytic_solution(x=x_data, y=y_data)
print('Trained weight: \n', Analytic_model.W.reshape(-1))
print()

# model evaluation
inference = Analytic_model.eval(x_data)

# Error calculation
error = RMSE(inference, y_data)
print('RMSE on Test Data : %.4f \n' % error)

"""
You should get results as:

Initial weight:
 [0. 0. 0. 0.]

Trained weight: 
 [ 0.76694866 20.82887592 54.13176192 76.77324443]

RMSE on Test Data : 0.3732
"""
#======================================================================================================
