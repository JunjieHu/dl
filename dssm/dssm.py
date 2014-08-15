import cPickle as pickle
import os
import sys
import time
import numpy as np

import theano
import theano.tensor as T
import parameters
from mlp import HiddenLayer

theano.config.openmp = True
batch_size = parameters.batch_size
n_layer1 = parameters.n_layer1
n_layer2 = parameters.n_layer2
n_layer3 = parameters.n_layer3
n_pos_neg = parameters.n_pos_neg
n_return_list = parameters.n_return_list


class Representation(object):
    def __init__(self, x, h_params):
        h_w1, h_b1, h_w2, h_b2, h_w3, h_b3 = h_params
        h_layer1 = HiddenLayer(x,h_w1,h_b1)
        h_layer2 = HiddenLayer(h_layer1.output,h_w2,h_b2)
        h_layer3 = HiddenLayer(h_layer2.output,h_w3,h_b3)
        
        self.output = h_layer3.output
        self.h_params = h_params
        
class train(object):
    def __init__(self, query, pos_neg_doc, h_params, learning_rate):
        
#        n_query   = query.shape[0]
#        self.h_params = h_params        
#        self.cost = 0
#        for i in range(n_query):
#            rep_query = T.vector()
#            rep_pos_neg_doc = T.matrix()
#            # get the representation of query and docs
#            rep_query   = Representation(query[i,:],self.h_params)
#            rep_pos_neg_doc = Representation(pos_neg_doc[i*n_pos_neg:(i+1)*n_pos_neg,:],self.h_params)
#            
#            # calculate the cosine similarity of query and docs
#            norm_q =T.TensorType(dtype='float32',broadcastable=[True],name='norm_q')
#            norm_q   = T.sqrt(T.sum(T.sqr(rep_query)))
#            norm_doc = T.sqrt(T.sum(T.sqr(rep_pos_neg_doc),1))
#            numerator = T.TensorType(dtype='float32',broadcastable=[True,True],name='num')
#            numerator = T.dot(rep_query,rep_pos_neg_doc.T)
#            pro =T.TensorType(dtype='float32',broadcastable=[True,True],name='pro')
#            pro = norm_q*norm_doc
#            cos = numerator/pro
#            
#            # get the softmax and -log(prob)
#            st = T.nnet.softmax(cos)
#            self.cost += -T.log(st[0][0])
        
        # update each query once at a time
        self.h_params = h_params        
        
        # get the representation of query and docs
        rep_query   = Representation(query,self.h_params).output
        rep_pos_neg_doc = Representation(pos_neg_doc,self.h_params).output
            
        # calculate the cosine similarity of query and docs
        self.norm_q = T.TensorType(dtype='float32',broadcastable=[True],name='norm_q')
        self.norm_q   = T.sqrt(T.sum(T.sqr(rep_query)))
        self.norm_doc = T.sqrt(T.sum(T.sqr(rep_pos_neg_doc),1))
        self.numerator = T.TensorType(dtype='float32',broadcastable=[True,True],name='num')
        self.numerator = T.dot(rep_query,rep_pos_neg_doc.T)
        self.pro = T.TensorType(dtype='float32',broadcastable=[True,True],name='pro')
        self.pro = self.norm_q*self.norm_doc
        self.cos = self.numerator/self.pro
            
        # get the softmax and -log(prob)
        self.st = T.nnet.softmax(self.cos)
        self.cost = -T.log(self.st[0][0])
        
        self.rep_query = rep_query
        self.rep_pos_neg_doc = rep_pos_neg_doc
        
        self.grad_h_params = T.grad(self.cost,self.h_params)
        updates = []
        for param, grad_param in zip(self.h_params,self.grad_h_params):
            updates.append((param, param - learning_rate*grad_param))
        self.updates = updates
        
            
            
def train_dssm(dataset='synthetics_1.npz'):
    # load data by JJ
    r = np.load(dataset)
    x = r['x']
    #query_id = r['query'].astype(int)
    sim_list = r['sim_list'].astype(int)   # [Q1, D_1,...,D_10]
    train_id = r['train_id'].astype(int)   # [Q1, D^+, D^-,D^-,D^-,D^-]
    test_id  = r['test_id'].astype(int)
    valid_id = r['valid_id'].astype(int)
    batch_size = parameters.batch_size
    learning_rate = parameters.learning_rate
   
    # get the real data
    train_query   = x[train_id[:,0],:]
    train_pos_neg = x[train_id[:,1:6].ravel(),:] 
    test_query   = x[test_id[:,0],:]
    test_pos_neg = x[test_id[:,1:6].ravel(),:] 
    valid_query   = x[valid_id[:,0],:]
    valid_pos_neg = x[valid_id[:,1:6].ravel(),:] 



    # compute number of minibatches for training, validation and testing
    n_train_batches = train_query.shape[0] / batch_size
    n_valid_batches = valid_query.shape[0] / batch_size
    n_test_batches = test_query.shape[0] / batch_size 
    
    ###############
    # Build Model #
    ###############
    T_query = T.matrix('query')
    T_pos_neg_doc = T.matrix('pos_neg_doc')
    h_params  = parameters.h_random_params() #loaddata('./h_parameter.pkl')

    train_handle = train(query=T_query, pos_neg_doc=T_pos_neg_doc, 
                          h_params=h_params, learning_rate=learning_rate )

    f = theano.function(inputs=[T_query,T_pos_neg_doc],outputs=[train_handle.rep_query,train_handle.rep_pos_neg_doc,
                        train_handle.norm_q,train_handle.norm_doc,train_handle.numerator,
                        train_handle.pro,train_handle.cos,train_handle.st,train_handle.cost],
                        updates=train_handle.updates
                        )
    
    ###############
    # Train Model #
    ###############
    startepoch=0
    for epoch in range(startepoch,100):
        print "epoch", epoch
        cnt = 0
        cost = 0.0
        for idx in range(n_train_batches):           
            # get the idx-th query and 1positive-4negative docs
            e_query =  train_query[idx*batch_size:(idx+1)*batch_size,:]
            e_pos_neg = train_pos_neg[idx*batch_size*n_pos_neg:(idx+1)*batch_size*n_pos_neg,:]

            result = f(e_query,e_pos_neg)
            cnt += 1
            cost += result[8]
            
            if cnt%100==0:
                print "save data" # save data
        print "Cost", cost
                  
if __name__ == '__main__':
    train_dssm()            
            
        