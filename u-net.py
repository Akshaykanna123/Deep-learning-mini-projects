import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model

X = np.random.rand(1,128,128,1)
y = (X > 0.5).astype(float)   # fake mask

inp = Input((128,128,1))

c1 = Conv2D(8,3,activation='relu',padding='same')(inp)
p1 = MaxPooling2D()(c1)

c2 = Conv2D(16,3,activation='relu',padding='same')(p1)

u1 = UpSampling2D()(c2)
c3 = Conv2D(8,3,activation='relu',padding='same')(u1)

out = Conv2D(1,1,activation='sigmoid')(c3)

model = Model(inp, out)
model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(X, y, epochs=3, verbose=0)

pred = model.predict(X)

plt.subplot(1,2,1)
plt.title("Input")
plt.imshow(X[0].reshape(128,128), cmap='gray')

plt.subplot(1,2,2)
plt.title("Segmented")
plt.imshow(pred[0].reshape(128,128), cmap='gray')
plt.show()
