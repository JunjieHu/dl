__author__ = 'Junjie HU'


import numpy
import theano

floatX = theano.config.floatX

batch_size = 1



learning_rate = 0.1

n_in = 10
n_layer1 = 10
n_layer2 = 10
n_layer3 = 5
n_pos_neg = 5
n_return_list = 9

def h_random_params():
    rng = numpy.random.RandomState(2014)
    
    h_w1 = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(n_in + n_layer1)),\
                        high=numpy.sqrt(6./(n_in + n_layer1)),size=(n_in,n_layer1)),dtype=floatX)
    h_b1 = numpy.zeros((n_layer1,), dtype=floatX)

    h_w2 = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(n_layer1 + n_layer2)),\
                        high=numpy.sqrt(6./(n_layer1 + n_layer2)),size=(n_layer1,n_layer2)),dtype=floatX)
    h_b2 = numpy.zeros((n_layer2,), dtype=floatX)

    h_w3 = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(n_layer2 + n_layer3)),\
                        high=numpy.sqrt(6./(n_layer2 + n_layer3)),size=(n_layer2,n_layer3)),dtype=floatX)
    h_b3 = numpy.zeros((n_layer3,), dtype=floatX)

    return [theano.shared(h_w1, borrow = True),theano.shared(h_b1, borrow = True),\
            theano.shared(h_w2, borrow = True),theano.shared(h_b2, borrow = True),\
            theano.shared(h_w3, borrow = True),theano.shared(h_b3, borrow = True)]
#
#def c_random_weights():
#
#   rng = numpy.random.RandomState(20166)
#
#
#   c_w0 = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(filter_shape0[0]+filter_shape0[1])),\
#                     high=numpy.sqrt(6./(filter_shape0[0]+filter_shape0[1])),size=filter_shape0),dtype=floatX)
#   c_b0 = numpy.zeros((filter_shape0[0],), dtype=floatX)
#
#   c_w1 = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(filter_shape1[0]+filter_shape1[1])),\
#                         high=numpy.sqrt(6./(filter_shape1[0]+filter_shape1[1])),size=filter_shape1),dtype=floatX)
#   c_b1 = numpy.zeros((filter_shape1[0],), dtype=floatX)
#
#   c_w2 = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(filter_shape2[0]+filter_shape2[1])),\
#                                 high=numpy.sqrt(6./(filter_shape2[0]+filter_shape2[1])),size=filter_shape2),dtype=floatX)
#   c_b2 = numpy.zeros((filter_shape2[0],), dtype=floatX)
#
#   c_w3 = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(filter_shape3[0]+filter_shape3[1])),\
#                                 high=numpy.sqrt(6./(filter_shape3[0]+filter_shape3[1])),size=filter_shape3),dtype=floatX)
#   c_b3 = numpy.zeros((filter_shape3[0],), dtype=floatX)
#   
#   return [theano.shared(c_w0, borrow = True),theano.shared(c_b0, borrow = True),\
#           theano.shared(c_w1, borrow = True),theano.shared(c_b1, borrow = True),\
#           theano.shared(c_w2, borrow = True),theano.shared(c_b2, borrow = True),\
#           theano.shared(c_w3, borrow = True),theano.shared(c_b3, borrow = True)]
#
#class Parameters:
#    """
#    Parameters used by the Model
#    """
#
#    def __init__(self):
#        self.embeddings_path = '../data_senti/wiki_embeddings.txt'
#        self.word2id = {}
#        self.words= []
#    def readEmbeddeing(self):
#      with open(self.embeddings_path,'r') as f:
#        line = f.readline()
#        vocab_size, embedding_size = line.strip('\n').split()
#        self.vocab_size = int(vocab_size)
#        self.embedding_size = int(embedding_size)
#        self.embeddings = numpy.asarray(numpy.random.rand(self.vocab_size, self.embedding_size),dtype=float)
#        self.embeddings = self.embeddings * 0
#        for i in range(self.vocab_size):
#          line = f.readline()
#          tmp_embedding = line.strip('\n').split()
#          self.words.append(tmp_embedding[0])
#          self.word2id[tmp_embedding[0]] = i
#          tmp_embedding = tmp_embedding[1:]
#          tmp_embedding = [float(elem) for elem in tmp_embedding]
#          self.embeddings[i] = tmp_embedding
#
#    def getEmbedding(self):
#        return self.embeddings
#
#    def getTrain(self):
#        return self.train
#
#    def getTest(self):
#        return self.test
#
#    def getWord2id(self):
#        return self.word2id
#
#if __name__ == "__main__":
#  para = Parameters()
#  para.readEmbeddeing()
#  print para.word2id['good']
