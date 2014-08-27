import theano.tensor as T
import theano
import numpy
def _dropout_from_layer(layer, p):
  random_seed = 1234
  rng = numpy.random.RandomState(random_seed)
  srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
  mask = srng.binomial(n=1, p=1-p, size=layer.shape)
  output = layer * T.cast(mask, theano.config.floatX)
  return output

