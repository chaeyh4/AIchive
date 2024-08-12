import numpy as np


class LogisticRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.zeros((self.num_features, 1))

    def train(self, x, y, batch_size, epochs, lr, optimizer):
        """
        N : # of training data
        D : # of features
        C : # of classes

        Inputs:
        x : (N, D), input data
        y : (N, )
        batch_size : (int) # of minibatch size
        epochs : (int) # of training epoch to execute
        lr : (float), learning rate
        optimizer : (Class) optimizer to use

        Returns:
        None

        Description:
        Given training data, hyperparameters and optimizer, execute training procedure.
        Weight should be updated by minibatch (not the whole data at a time)
        The procedure for one epoch is as follows:
        - For each minibatch
            - Compute the probability of each class for data and the loss
            - Compute the gradient of weight
            - Update weight using optimizer

        * loss of one epoch = refer to the loss function in the instruction.
        """

        num_data, num_feat = x.shape
        num_batches = int(np.ceil(num_data / batch_size))

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            
            for b in range(num_batches):
                ed = min(num_data, (b + 1) * batch_size)
                batch_x = x[b * batch_size: ed]
                batch_y = y[b * batch_size: ed]

                prob, loss = self.forward(batch_x, batch_y)
                
                grad = self.compute_grad(batch_x, batch_y, self.W, prob)

                # Update Weights
                self.W = optimizer.update(self.W, grad, lr)

                epoch_loss += loss

    def forward(self, x, y):
        """
        N : # of minibatch data
        D : # of features

        Inputs:
        x : (N, D), input data 
        y : (N, ), label for each data

        Returns:
        logits: (N, 1), logit for N data
        loss : float, loss for N input

        Description:
        Given N data and their labels, compute probability distribution and loss.

        Hint:
        For the log function, add epsilon (eps) for numerical stability in log.
        e.g., np.log(logits) -> np.log(logits + eps)
        """

        num_data, num_feat = x.shape

        y = np.expand_dims(y, axis=1)
        
        logits = None
        loss = 0.0
        eps = 1e-10

        # ========================= EDIT HERE ========================
        fx = np.dot(x, self.W)
        logits = self._sigmoid(fx)
        loss = (-1/num_data) * np.sum(y * np.log(logits + eps) + (1 - y) * np.log(1 - logits + eps))
        # ============================================================

        return logits, loss
    
    def compute_grad(self, x, y, weight, logit):
        """
        N : # of minibatch data
        D : # of features

        Inputs:
        x : (N, D), input data
        y : (N, ), label for each data
        weight : (D, 1), Weight matrix of classifier
        logit: (N, 1), logits for N data

        Returns:
        the gradient of weight: (D, 1), Gradient of weight to be applied (dL/dW)

        Description:
        Given input, label, weight, logit, compute the gradient of weight.
        """

        num_data, num_feat = x.shape
        
        y = np.expand_dims(y, axis=1)
        
        grad_weight = np.zeros_like(weight)

        # ========================= EDIT HERE ========================
        grad_weight = (1 / num_data) * np.dot(x.T, logit - y)
        # ============================================================

        return grad_weight
    
    def _sigmoid(self, x):
        """
        Inputs:
        x : (N, C), score before sigmoid

        Returns:
        sigmoid : (same shape with x), applied sigmoid.

        Description:
        Given an input x, apply the sigmoid function.
        """

        sigmoid = None

        # ========================= EDIT HERE ========================
        sigmoid = 1 / (1 + np.exp(-x))
        # ============================================================

        return sigmoid

    def eval(self, x, threshold=0.5):
        """
        Inputs:
        x : (N, D), input data

        Returns:
        pred : (N, ), predicted label for N test data
        prob : (N, 1), predicted logit for N test data

        Description:
        Given N test data, compute logits and make predictions for each data with the given threshold.
        """

        pred = None
        prob = None

        # ========================= EDIT HERE ========================
        logits = np.dot(x, self.W)
        prob = self._sigmoid(logits)
        pred = []
        for i in prob:
            if i >= threshold:
                pred.append(1)
            else:
                pred.append(0)
        pred = np.array(pred)
        # ============================================================

        return pred, prob
