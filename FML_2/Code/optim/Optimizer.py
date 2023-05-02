import numpy as np


class SGD:
    def __init__(self):
        pass

    def update(self, w, grad, lr):
        return w - lr * grad