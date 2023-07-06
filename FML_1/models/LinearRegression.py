import numpy as np

class LinearRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.zeros((self.num_features, 1))

    def numerical_solution(self, x, y, epochs, batch_size, lr, optim, batch_gradient=False):

        """
        The numerical solution of Linear Regression
        Train the model for 'epochs' times with minibatch size of 'batch_size' using gradient descent.
        (TIP : if the dataset size is 10, and the minibatch size is set to 3, corresponding minibatch size should be 3, 3, 3, 1)

        [Inputs]
            x : input for linear regression. Numpy array of (N, D)
            y : label of data x. Numpy array of (N, )
            epochs : epochs.
            batch_size : size of the batch.
            lr : learning rate.
            batch_gradient : If true, use batch gradient descent method; otherwise, use minibatch stochastic gradient descent method(Default).
            optim : optimizer.
        [Output]
            None
        """

        self.W = self.W.reshape(-1)
        num_data = len(x)
        num_batch = int(np.ceil(num_data / batch_size))
        
        for epoch in range(epochs):
            
            if batch_gradient:
                loss_vector = np.dot(x, self.W) - y
                grad = np.dot(loss_vector, x) / num_data
                self.W = optim.update(self.W, grad, lr)

            else:
                for batch_index in range(num_batch):
                    
                    batch_x = x[batch_index*batch_size:(batch_index+1)*batch_size]
                    batch_y = y[batch_index*batch_size:(batch_index+1)*batch_size]

                    num_samples_in_batch = len(batch_x)

                    loss_vector = np.dot(batch_x, self.W) - batch_y
                    grad = np.dot(loss_vector, batch_x) / num_samples_in_batch

                    self.W = optim.update(self.W, grad, lr)

    def analytic_solution(self, x, y):
        """
        The analytic solution of Linear Regression
        Train the model using the analytic solution.

        [Inputs]
            x : input for linear regression. Numpy array of (N, D)
            y : label of data x. Numpy array of (N, )

        [Output]
            None

        [Hints]
            1. Use np.transpose for transposing a matrix.
            2. Use np.linalg.inv for making a inverse matrix.
            3. Use np.dot for performing a dot product between two matrices.
        """
        # ==== EDIT HERE ====
        A = np.linalg.inv(np.dot(np.transpose(x),x))
        b = np.dot(np.transpose(x), y)
        self.W = np.dot(A, b)
        # ===================

    def eval(self, x):
        pred = None

        """
        Evaluation Function
        [Input]
            x : input for linear regression. Numpy array of (N, D)

        [Outputs]
            pred : prediction for 'x'. Numpy array of (N, )
        """

        pred = np.dot(x, self.W)

        return pred
