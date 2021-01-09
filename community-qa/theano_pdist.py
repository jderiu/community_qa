import theano
import theano.tensor as T
import numpy as np
import keras.backend as K
from scipy.spatial.distance import cdist

rng = np.random.RandomState(42)
d = 20 # dimension
nX = 10
nY = 30

x = np.random.rand(1, 121, 52).astype('float32')
y = np.random.rand(1, 200, 52).astype('float32')

#x = np.asarray([[ [1,2,3], [1,2,1] ]]).astype('float32')
#y = np.asarray([[ [1,2,3], [1,2,3], [1,2,3] ]]).astype('float32')

print cdist(x[0], y[0])

X = T.tensor3('X', dtype='float32')
Y = T.tensor3('Y', dtype='float32')

x_square = K.square(X)
y_square = K.square(Y)

x_sq_sum = K.repeat(K.sum(x_square, axis=-1), n=y_square.shape[1])
y_sq_sum = K.repeat(K.sum(y_square, axis=-1), n=x_square.shape[1])

dot = K.batch_dot(X, K.permute_dimensions(Y, (0, 2, 1)), axes=(2, 1))

squared_euclidean_distances = K.sqrt(K.permute_dimensions(x_sq_sum, (0, 2, 1)) + y_sq_sum - 2*dot)

f_x = theano.function([X, Y], x_sq_sum)
f_y = theano.function([Y, X], y_sq_sum)
f_xy = theano.function([X, Y], dot)
f_euclidean = theano.function([X, Y], squared_euclidean_distances)

d_eucl = f_x(x,y)
print d_eucl.shape
d_eucl = f_y(y,x)
print d_eucl.shape
d_eucl = f_xy(x, y)
print d_eucl.shape
d_eucl = f_euclidean(x,y)
print d_eucl

print np.mean(cdist(x[0], y[0]) - d_eucl)