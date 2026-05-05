import numpy as np
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.datasets import mnist
import keras.backend as K

(X,_),_ = mnist.load_data()
X = X.reshape(-1,784)/255.0

inp = Input(shape=(784,))
h = Dense(32, activation='relu')(inp)

z_mean = Dense(2)(h)
z_log = Dense(2)(h)

def samp(args):
    m, l = args
    eps = K.random_normal(shape=(K.shape(m)[0],2))
    return m + K.exp(0.5*l)*eps

z = Lambda(samp)([z_mean, z_log])

out = Dense(784, activation='sigmoid')(z)

model = Model(inp, out)

loss = K.sum(K.binary_crossentropy(inp, out), axis=-1) \
       -0.5*K.sum(1+z_log-K.square(z_mean)-K.exp(z_log), axis=-1)

model.add_loss(K.mean(loss))
model.compile(optimizer='adam')

model.fit(X, epochs=5, batch_size=128)
