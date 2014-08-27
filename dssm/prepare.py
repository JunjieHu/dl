import numpy
import parameters
import theano
import theano.tensor as T
import scipy.spatial
import random
import time
from w2v_embed_record import WordEmbedding
'''
 this is the load data file
'''

class prepareData(object):
  def __init__(self):
    we = pickle.load(open('w2v_embed.pkl','rb'))
    


'''
class prepareData(object):
  def __init__(self):
    para = parameters.Parameters()
    para.readEmbeddeing()
    self.embeddings = para.getEmbedding()
    self.word2id = para.getWord2id()
    self.train_file = '../data_snipet/train_snip.txt'
    self.train_label = '../data_snipet/train_label.txt'
    self.test_file = '../data_snipet/test_snip.txt'
    self.test_label = '../data_snipet/test_label.txt'
    self.sen_len = 30
    
    self.train_inst = []
    self.test_inst = []
    self.unknown = self.word2id["unknown"]
    self.embedding_size = para.embedding_size
    i = 0     
    with open(self.train_file,'r') as fin_train:
      with open(self.train_label,'r') as fin_label:
        for i in range(10060):
          line1 = fin_train.readline()
          line2 = fin_label.readline()
          line1 = line1.lower()

          words = line1.split()
          label = line2.split()
          self.train_inst.append((words,numpy.int32(label[0])-1))
 
    with open(self.test_file,'r') as fin_test:
      with open(self.test_label,'r') as fin_label:
        for i in range(2280):
          line1 = fin_test.readline()
          line2 = fin_label.readline()
          line1 = line1.lower()
          
          words = line1.split()
          label = line2.split()
          self.test_inst.append((words,numpy.int32(label[0])-1))
    
    self.allZero = numpy.zeros([50],dtype = float)
    self.block_size = 10

  def fromtext2vector(self,sentence1):
    sentence_vec = numpy.asarray(numpy.zeros([self.sen_len,self.embedding_size],dtype='float32'),dtype = 'float32')
    
    len1 = min(len(sentence1),29) 
    for k in range(len1):
      index = self.word2id.get(sentence1[k],self.unknown)
      sentence_vec[k] = self.embeddings[index]
    sentence_vec =  sentence_vec.reshape(1,self.sen_len, self.embedding_size)
    return sentence_vec

  def generate_batch_from_sentence(self, batch_size):
    res = []
    k = 0
    index = 0
    batch = numpy.asarray(numpy.zeros([batch_size,1,self.sen_len,self.embedding_size],dtype='float32'),dtype = 'float32')
    label = numpy.asarray(numpy.zeros(batch_size,dtype='int32'),dtype='int32')


    for i in range(10060):
      select_inst = random.randint(0,10059)  
      batch[index] = self.fromtext2vector(self.train_inst[select_inst][0])
      label[index] = self.train_inst[select_inst][1]
      index = index + 1

      if index == batch_size:
        res.append((batch,label))
        index = 0
        k = k + 1
        batch= numpy.asarray(numpy.zeros([batch_size,1,self.sen_len,self.embedding_size],dtype='float32'),dtype = 'float32')
        label = numpy.asarray(numpy.zeros(batch_size,dtype='int32'),dtype='int32') 
      if k==self.block_size:
        yield res
        res = []
        k = 0


  def generate_test_from_sentence(self, batch_size):
    res = []
    k = 0
    index = 0
    batch = numpy.asarray(numpy.zeros([batch_size,1,self.sen_len,self.embedding_size],dtype='float32'),dtype = 'float32')
    label = numpy.asarray(numpy.zeros(batch_size,dtype='int32'),dtype='int32')


    for i in range(2280):
      batch[index] = self.fromtext2vector(self.test_inst[i][0])
      label[index] = self.test_inst[i][1]
      index = index + 1

      if index == batch_size:
        res.append((batch,label))
        index = 0
        k = k + 1
        batch= numpy.asarray(numpy.zeros([batch_size,1,self.sen_len,self.embedding_size],dtype='float32'),dtype = 'float32')
        label = numpy.asarray(numpy.zeros(batch_size,dtype='int32'),dtype='int32') 
      if k==self.block_size:
        yield res
        res = []
        k = 0

if __name__ =="__main__":
  prepare = prepareData()
  start = time.time()
  k = prepare.generate_test_from_sentence(2)
  j = 0
  for item in k:
    tmp = item[0]
    #for i in range(2):
     # for l in range(2):
      #  print item[0][0][i][l]
  print 'beautiful!','it takes',time.time()-start

'''










