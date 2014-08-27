import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from ReLU import ReLU
class LeNetConvPoolLayer(object):
  def __init__(self, input, filter_shape=None, image_shape=None,W=None, b=None, poolsize=(3, 1)):
    assert image_shape[1] == filter_shape[1]
    self.W = W
    self.b = b 
    tmp = numpy.ones((filter_shape[0],), dtype=theano.config.floatX)
    tmp = -tmp*10000
    self.test = theano.shared(value=tmp, borrow=True)
    conv_out = conv.conv2d(input=input, filters=self.W, filter_shape=filter_shape, image_shape=image_shape)
    conv_out2 = T.where(T.neq(conv_out,0), conv_out, self.test.dimshuffle('x', 0, 'x', 'x'))
    pooled_out = downsample.max_pool_2d(conv_out2,ds=poolsize, ignore_border=True)
    pooled_out2 = T.where(T.neq(pooled_out, -10000) ,pooled_out, -self.b.dimshuffle('x', 0, 'x', 'x')) 
    self.output = ReLU(pooled_out2 + self.b.dimshuffle('x', 0, 'x', 'x'))
    #self.output = T.nnet.sigmoid(pooled_out2 + self.b.dimshuffle('x', 0, 'x', 'x'))
    self.params = [self.W, self.b]

