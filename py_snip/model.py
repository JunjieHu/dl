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

dotrain = 0
startepoch = 0
para_file = ''

i = 1
while(i < len(sys.argv)):
   arg = sys.argv[i]
   if arg == '-train':
      dotrain = 1
      learning_rate = float(sys.argv[i+1])
   elif arg == '-startepoch':
     startepoch = int(sys.argv[i+1])
   else:
     pass
   i = i + 2

cmd = 'mkdir ' + 'sigmoid_para_' + 'learning_rate' + str(learning_rate)
base_para_path = 'sigmoid_para_' + 'learning_rate' + str(learning_rate) + '/'
os.system(cmd)


class MatchModel(object):
  def __init__(self,input=None, y=None, Cparams=None, Mparams=None):

    c_w0, c_b0, c_w1, c_b1, c_w2, c_b2, c_w3, c_b3 = Cparams
    m_w1, m_b1, o_w1, o_b1 = Mparams

    c_layer0 = LeNetConvPoolLayer(input=input, filter_shape=filter_shape0, image_shape=image_shape0,W=c_w0, b=c_b0, poolsize=poolsize0)
    
    c_layer1 = LeNetConvPoolLayer(input=c_layer0.output, filter_shape=filter_shape1, image_shape=image_shape1,W=c_w1, b=c_b1, poolsize=poolsize1)
    
    c_layer2 = LeNetConvPoolLayer(input=c_layer1.output, filter_shape=filter_shape2, image_shape=image_shape2,W=c_w2, b=c_b2, poolsize=poolsize2)
    c_layer3 = LeNetConvPoolLayer(input=c_layer2.output, filter_shape=filter_shape3, image_shape=image_shape3,W=c_w3, b=c_b3, poolsize=poolsize3)
   
    m_input = c_layer3.output
    m_input = m_input.flatten(2)
    m_layer1 = HiddenLayer(m_input, W=m_w1, b=m_b1)
    s_layer = LogisticRegression(m_layer1.output, W=o_w1, b=o_b1)
    self.cost= s_layer.negative_log_likelihood(y)

class train(object):
  def __init__(self, P_input, P_y, Cparams, Mparams, learning_rate):
    self.P_input = P_input
    self.P_y = P_y
    self.Cparams = Cparams
    self.Mparams = Mparams

    Pmatch=MatchModel(input=self.P_input, y=P_y, Cparams=self.Cparams, Mparams=self.Mparams)
    self.cost =  Pmatch.cost.mean()

    self.gMparams = T.grad(self.cost, self.Mparams)
    updates = []
    for param, gMparam in zip(self.Mparams, self.gMparams):
      updates.append((param, param - learning_rate*gMparam))
    
    #self.gCparams = T.grad(self.cost, self.Cparams)
    #for param, gCparam in zip(self.Cparams, self.gCparams):
     # updates.append((param, param - 0.1*learning_rate*gCparam))
    
    self.updates = updates

  def storedata(self,index):
    ofile=open(base_para_path+str(index)+"Mparas.pkl",'wb')
    pickle.dump(self.Mparams, ofile)
    ofile.close()

  def c_storedata(self,index):
    ofile=open(base_para_path+str(index)+"rCparas.pkl",'wb')
    pickle.dump(self.Cparams, ofile)
    ofile.close()

def loaddata(filename):
  ofile = open(filename,'rb')
  params=pickle.load(ofile)
  ofile.close()
  return params

def train_worker():
  P_input = T.tensor4(dtype=theano.config.floatX)
  P_y = T.vector(name ='P_y', dtype='int32')
  Cparams = parameters.c_random_weights()#loaddata('./Cparams.pkl') 
  Mparams = parameters.random_weights()

  train_handle = train(P_input=P_input, P_y=P_y, Cparams=Cparams, Mparams=Mparams, learning_rate=learning_rate)
  f = theano.function(inputs=[P_input, P_y],outputs=train_handle.cost, updates=train_handle.updates)

  prep = prepare.prepareData()
  block_size = prep.block_size

  for epoch in range(startepoch,100):
    print "epoch", epoch
    cnt = 0.0
    cost = 0.0
    item = prep.generate_batch_from_sentence(batch_size)
    for pair in item :
      for i in range(block_size):
          a = f(pair[i][0],pair[i][1])
          cost = cost + a
          cnt = cnt + 1
          sys.stdout.flush()
          sys.stdout.write(str(cnt)+'\r')
          #if cnt*batch_size%10000 == 0:
            #train_handle.storedata(cnt)
    train_handle.storedata(epoch)
    train_handle.c_storedata(epoch)
    print cost


if __name__ == '__main__':
  if dotrain:
    print 'training...'
    train_worker()

