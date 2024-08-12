import numpy as np

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    _exp = np.exp(z)
    _sum = np.sum(_exp, axis=1, keepdims=True)
    sm = _exp / _sum
    return sm

class ReLU:

    """
    ReLU Function. ReLU(x) = max(0, x)
    Implement forward & backward path of ReLU.
    """
    def __init__(self):
        self.zero_mask = None
    def forward(self, z):
        """
        ReLU Forward.
        ReLU(x) = max(0, x)

        z --> (ReLU) --> out

        [Inputs]
            z : ReLU input in any shape.

        [Outputs]
            out : ReLU(z).
        """
        out = None

        self.zero_mask = z < 0
        z[self.zero_mask] = 0
        out = z
        
        return out

    def backward(self, d_prev):
        """
        ReLU Backward.

        z --> (ReLU) --> out
        dz <-- (dReLU) <-- d_prev(dL/dout)

        [Inputs]
            d_prev : Gradients until now.
            d_prev = dL/dk, where k = ReLU(z).

        [Outputs]
            dz : Gradients w.r.t. ReLU input z.
        """
        dz = None
        # =============== EDIT HERE ===============
        dz = d_prev.copy()
        dz[self.zero_mask] = 0
        # =========================================

        return dz

class Sigmoid:
    def __init__(self):
        self.out = None
    def forward(self, z):
        """
        Sigmoid Forward.

        z --> (Sigmoid) --> self.out

        [Inputs]
            z : Sigmoid input in any shape.

        [Outputs]
            self.out : Sigmoid(z).
        """

        self.out = None
        self.out = 1 / (1 + np.exp(-z))
        
        return self.out

    def backward(self, d_prev):
        """
        Sigmoid Backward.

        z --> (Sigmoid) --> self.out
        dz <-- (dSigmoid) <-- d_prev(dL/d self.out)

        [Inputs]
            d_prev : Gradients until now.

        [Outputs]
            dz : Gradients w.r.t. Sigmoid input z.
        """

        dz = None
        # =============== EDIT HERE ===============
        derivative = self.out * (1 - self.out)
        dz = d_prev * derivative
        # =========================================
        return dz

class SigmoidCELayer:
    def __init__(self, num_hidden_2, num_outputs):
        limit = np.sqrt(2 / float(num_hidden_2))
        self.W = np.random.normal(0.0, limit, size=(num_hidden_2, num_outputs))

        self.b = np.zeros(num_outputs)

        self.dW = None
        self.db = None

        self.x = None
        self.y = None
        self.y_hat = None

        self.loss = None

        self.sigmoid = Sigmoid()

    def forward(self, x, y):
        """
        Sigmoid output layer forward
        - Make prediction
        - Calculate loss

        """
        self.y_hat = self.predict(x)
        self.y = y
        self.x = x

        self.loss = self.binary_ce_loss(self.y_hat, self.y)

        return self.loss

    def binary_ce_loss(self, y_hat, y):
        """
        Calcualte "Binary cross-entropy loss"
        Add 'eps' for stability inside log function.

        [Inputs]
            y_hat : Prediction
            y : Label

        [Outputs]
            loss value
        """
        eps = 1e-10
        bce_loss = None

        bce_loss = -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))

        return bce_loss

    def predict(self, x):
        """
        Make prediction in probability. (Not 0 or 1 label!!)

        [Inputs]
            x : input data

        [Outputs]
            y_hat : Prediction
        """
  
        z = np.matmul(x, self.W) + self.b
        y_hat = self.sigmoid.forward(z)
        
        return y_hat

    def backward(self, d_prev=1):
        """
        Calculate gradients of input (x), W, b of this layer.
        Save self.dW, self.db to update later.

        x and (W & b) --> z -- (activation) --> y_hat --> Loss
        dx and (dW & db) <-- dz <-- (activation) <-- dy_hat <-- Loss

        [Inputs]
            d_prev : Gradients until here. (Always 1 since its output layer)

        [Outputs]
            dx : Gradients of output layer input x (Not MLP input x!)
        """
        batch_size = self.y.shape[0]
        d_z = (self.y_hat - self.y.reshape(-1, 1)) / batch_size
        d_prev = 1

        d_sigmoid = None
        # =============== EDIT HERE ===============
        """
        you should calcualte grandient of sigmoid layer
        !!!!self.sigmoid!!!! 
        """
        d_sigmoid = self.sigmoid.backward(d_prev)
        # =========================================
        d_z = d_sigmoid * d_z

        self.dW = self.x.T.dot(d_z)
        self.db = np.sum(d_z, axis=0)
        dx = np.matmul(self.W, d_z.T).T
        return dx


class Linear:
    def __init__(self, num_hidden_1, num_hidden_2):
        limit = np.sqrt(2 / float(num_hidden_1))
        self.W = np.random.normal(0.0, limit, size=(num_hidden_1, num_hidden_2))
        self.b = np.zeros(num_hidden_2)

        self.dW = None
        self.db = None

    def forward(self, x):
        """
        Linear layer forward
        - Feed forward
        - Apply activation function you implemented above.

        [Inputs]
           x : Input data (N, D)

        [Outputs]
            self.out : Output of Linear layer. Hidden. (N, H)
        """

        self.x = x
        self.out = np.matmul(self.x, self.W) + self.b

        return self.out

    def backward(self, d_prev):
        """
        Linear layer backward
        x and (W & b) --> z -- (activation) --> hidden
        dx and (dW & db) <-- dz <-- (activation) <-- hidden

        - Backward of activation
        - Gradients of W, b

        [Inputs]
            d_prev : Gradients until now.

        [Outputs]
            dx : Gradients of input x
        """
        dx = None
        self.dW = None
        self.db = None
        # =============== EDIT HERE ===============
        dx = np.matmul(d_prev, self.W.T)
        self.dW = np.matmul(self.x.T, d_prev)
        self.db = np.sum(d_prev, axis=0)
        # =========================================

        return dx