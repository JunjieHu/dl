import cPickle as pickle
import os
import sys
import time

import theano
import theano.tensor as T
from LeNetConvPoolLayer import LeNetConvPoolLayer
from mlp import HiddenLayer
import parameters
import prepare
from dropout import _dropout_from_layer
from logistic import LogisticRegression
theano.config.openmp = True
#theano.config.floatX = 'float32'
batch_size = parameters.batch_size
filter_shape0 = parameters.filter_shape0
filter_shape1 = parameters.filter_shape1
filter_shape2 = parameters.filter_shape2
filter_shape3 = parameters.filter_shape3

poolsize0 = parameters.poolsize0
poolsize1 = parameters.poolsize1
poolsize2 = parameters.poolsize2
poolsize3 = parameters.poolsize3

image_shape0 = parameters.image_shape0
image_shape1 = parameters.image_shape1
image_shape2 = parameters.image_shape2
image_shape3 = parameters.image_shape3

alpha = parameters.alpha
learning_rate = parameters.learning_rate


class MatchModel(object):
  def __init__(self,input=None, Cparams=None, Mparams=None):

    c_w0, c_b0, c_w1, c_b1, c_w2, c_b2, c_w3, c_b3 = Cparams
    m_w1, m_b1, o_w1, o_b1 = Mparams

    c1_layer0 = LeNetConvPoolLayer(input=input, filter_shape=filter_shape0, image_shape=image_shape0,W=c_w0, b=c_b0, poolsize=poolsize0)
    
    c1_layer1 = LeNetConvPoolLayer(input=c1_layer0.output, filter_shape=filter_shape1, image_shape=image_shape1,W=c_w1, b=c_b1, poolsize=poolsize1)
    
    c1_layer2 = LeNetConvPoolLayer(input=c1_layer1.output, filter_shape=filter_shape2, image_shape=image_shape2,W=c_w2, b=c_b2, poolsize=poolsize2)
    
    c1_layer3 = LeNetConvPoolLayer(input=c1_layer2.output, filter_shape=filter_shape3, image_shape=image_shape3,W=c_w3, b=c_b3, poolsize=poolsize3)

    m_input = c1_layer3.output
    m_input = m_input.flatten(2)
    m_layer1 = HiddenLayer(m_input, W=m_w1, b=m_b1)
    s_layer = LogisticRegression(m_layer1.output, W=o_w1, b=o_b1)
    #self.y_pred= s_layer.getlabel()
    self.y_pred= c1_layer3.output.flatten(1)
    #self.output = c1_layer3.output

class test(object):
  def __init__(self, P_input, Cparams, Mparams):
    self.P_input = P_input
    self.Cparams = Cparams
    self.Mparams = Mparams
    Pmatch=MatchModel(input=self.P_input, Cparams=self.Cparams, Mparams=self.Mparams)
    self.output =  Pmatch.y_pred

def loaddata(filename):
  ofile = open(filename,'rb')
  params=pickle.load(ofile)
  ofile.close()
  return params

def test_worker():
  fout = open('test_label.txt','w')
  P_input = T.tensor4(dtype=theano.config.floatX)
  
  Cparams = loaddata('./para_learning_rate0.01/0rCparas.pkl') 
  Mparams = loaddata('./para_learning_rate0.01/12Mparas.pkl')
  test_handle = test(P_input=P_input, Cparams=Cparams, Mparams=Mparams)
  f = theano.function(inputs=[P_input],outputs=test_handle.output, updates=[])
  
  right = 0.0
  cnt = 0.0

  prep = prepare.prepareData()
  block_size = prep.block_size
  item = prep.generate_test_from_sentence(1)
  for pair in item :
    for i in range(block_size):
      a = f(pair[i][0])
      print a
      #fout.write(str(a.mean()))
      #fout.write('\n')
      cnt = cnt + 1
      if pair[i][1].mean()==a.mean():
        right = right + 1.0
      #print right/cnt

if __name__ == '__main__':
  test_worker()

