import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten
from keras.datasets import mnist

(X,_),_ = mnist.load_data()
X = X / 255.0

low = X[:,::2,::2]
low = low.reshape(-1,196)
high = X.reshape(-1,784)

G = Sequential([
    Dense(256, input_dim=196, activation='relu'),
    Dense(784, activation='sigmoid')
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
    idx = np.random.randint(0, X.shape[0], 32)
    lr = low[idx]
    hr = high[idx]

    fake = G.predict(lr, verbose=0)

    D.train_on_batch(hr, np.ones((32,1)))
    D.train_on_batch(fake, np.zeros((32,1)))

    GAN.train_on_batch(lr, np.ones((32,1)))

test = low[0].reshape(1,196)
gen = G.predict(test, verbose=0).reshape(28,28)

plt.imshow(gen, cmap='gray')
plt.title("Generated High-Res")
plt.axis('off')
plt.show()
