import numpy
import theano
import theano.tensor as T
#from ReLU import ReLU
class HiddenLayer(object):
  def __init__(self,input, W=None, b=None):
    self.input = input
    self.W =W
    self.b =b
    self.output=T.nnet.sigmoid(T.dot(self.input, self.W)+self.b)
    self.params=[self.W, self.b]
