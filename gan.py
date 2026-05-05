import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist

(X,_),_ = mnist.load_data()
X = (X - 127.5) / 127.5
X = X.reshape(-1,784)

G = Sequential([
    Dense(128, input_dim=100, activation='relu'),
    Dense(784, activation='tanh')
])

D = Sequential([
    Dense(128, input_dim=784, activation='relu'),
    Dense(1, activation='sigmoid')
])
D.compile(loss='binary_crossentropy', optimizer='adam')

D.trainable = False
GAN = Sequential([G, D])
GAN.compile(loss='binary_crossentropy', optimizer='adam')

for i in range(500):
    noise = np.random.normal(0,1,(32,100))
    fake = G.predict(noise, verbose=0)
    real = X[np.random.randint(0, X.shape[0], 32)]

    D.train_on_batch(real, np.ones((32,1)))
    D.train_on_batch(fake, np.zeros((32,1)))

    noise = np.random.normal(0,1,(32,100))
    GAN.train_on_batch(noise, np.ones((32,1)))

noise = np.random.normal(0,1,(1,100))
img = G.predict(noise, verbose=0).reshape(28,28)

plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()
