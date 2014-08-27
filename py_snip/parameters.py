__author__ = 'Baotian HU'


import numpy
import theano

floatX = theano.config.floatX

batch_size = 1

nkerns=[100, 150,300, 400]
filter_shape0=(nkerns[0],1,3,50)
filter_shape1=(nkerns[1],nkerns[0],3,1)
filter_shape2=(nkerns[2],nkerns[1],3,1)
filter_shape3=(nkerns[3],nkerns[2],2,1)

poolsize0 = (2,1)
poolsize1 = (2,1)
poolsize2 = (2,1)
poolsize3 = (1,1)

image_shape0 = (batch_size,1,30,50)
image_shape1 = (batch_size,nkerns[0],14,1)
image_shape2 = (batch_size,nkerns[1],6,1)
image_shape3 = (batch_size,nkerns[2],2,1)

hidden_out=300
hidden_in = filter_shape3[0]
learning_rate = 0.001
alpha = 0.5


def random_weights():

   rng = numpy.random.RandomState(2014)

   m_w1 = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(hidden_in + hidden_out)),\
                        high=numpy.sqrt(6./(hidden_in+hidden_out)),size=(hidden_in,hidden_out)),dtype=floatX)
   m_b1 = numpy.zeros((hidden_out,), dtype=floatX)

   o_w1 = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(hidden_out+8)),\
                        high=numpy.sqrt(6./(hidden_out+8)),size=(hidden_out,8)),dtype=floatX)
   o_b1 = numpy.zeros((8,), dtype=floatX)

   return [theano.shared(m_w1, borrow = True),theano.shared(m_b1, borrow = True),\
           theano.shared(o_w1, borrow = True),theano.shared(o_b1, borrow = True)]

def c_random_weights():

   rng = numpy.random.RandomState(20166)


   c_w0 = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(filter_shape0[0]+filter_shape0[1])),\
                     high=numpy.sqrt(6./(filter_shape0[0]+filter_shape0[1])),size=filter_shape0),dtype=floatX)
   c_b0 = numpy.zeros((filter_shape0[0],), dtype=floatX)

   c_w1 = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(filter_shape1[0]+filter_shape1[1])),\
                         high=numpy.sqrt(6./(filter_shape1[0]+filter_shape1[1])),size=filter_shape1),dtype=floatX)
   c_b1 = numpy.zeros((filter_shape1[0],), dtype=floatX)

   c_w2 = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(filter_shape2[0]+filter_shape2[1])),\
                                 high=numpy.sqrt(6./(filter_shape2[0]+filter_shape2[1])),size=filter_shape2),dtype=floatX)
   c_b2 = numpy.zeros((filter_shape2[0],), dtype=floatX)

   c_w3 = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(filter_shape3[0]+filter_shape3[1])),\
                                 high=numpy.sqrt(6./(filter_shape3[0]+filter_shape3[1])),size=filter_shape3),dtype=floatX)
   c_b3 = numpy.zeros((filter_shape3[0],), dtype=floatX)
   
   return [theano.shared(c_w0, borrow = True),theano.shared(c_b0, borrow = True),\
           theano.shared(c_w1, borrow = True),theano.shared(c_b1, borrow = True),\
           theano.shared(c_w2, borrow = True),theano.shared(c_b2, borrow = True),\
           theano.shared(c_w3, borrow = True),theano.shared(c_b3, borrow = True)]

class Parameters:
    """
    Parameters used by the Model
    """

    def __init__(self):
        self.embeddings_path = '../data_senti/wiki_embeddings.txt'
        self.word2id = {}
        self.words= []
    def readEmbeddeing(self):
      with open(self.embeddings_path,'r') as f:
        line = f.readline()
        vocab_size, embedding_size = line.strip('\n').split()
        self.vocab_size = int(vocab_size)
        self.embedding_size = int(embedding_size)
        self.embeddings = numpy.asarray(numpy.random.rand(self.vocab_size, self.embedding_size),dtype=float)
        self.embeddings = self.embeddings * 0
        for i in range(self.vocab_size):
          line = f.readline()
          tmp_embedding = line.strip('\n').split()
          self.words.append(tmp_embedding[0])
          self.word2id[tmp_embedding[0]] = i
          tmp_embedding = tmp_embedding[1:]
          tmp_embedding = [float(elem) for elem in tmp_embedding]
          self.embeddings[i] = tmp_embedding

    def getEmbedding(self):
        return self.embeddings

    def getTrain(self):
        return self.train

    def getTest(self):
        return self.test

    def getWord2id(self):
        return self.word2id

if __name__ == "__main__":
  para = Parameters()
  para.readEmbeddeing()
  print para.word2id['good']
