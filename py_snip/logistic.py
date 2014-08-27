import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
import parameters
import scipy.spatial


class LogisticRegression(object):
    def __init__(self, input, W=None, b=None):

        self.W = W
        self.b = b
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
        self.L1 = T.sum(abs(self.W))+ T.sum(abs(self.b))
        self.L2 = T.sum(self.b ** 2) + T.sum(self.b**2)
    
    def getlabel(self):
        return self.y_pred
    
    def negative_log_likelihood(self,y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
