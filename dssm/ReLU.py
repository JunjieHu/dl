import theano.tensor as T
def ReLU(x):
    y = T.maximum(0.0, x)
    return y
