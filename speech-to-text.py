import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense

X = np.random.rand(50,5,10)
y = np.random.rand(50,5,15)

enc_in = Input(shape=(5,10))
_, h, c = LSTM(16, return_state=True)(enc_in)

dec_in = Input(shape=(5,15))
dec = LSTM(16, return_sequences=True)(dec_in, initial_state=[h,c])
out = Dense(15, activation='softmax')(dec)

model = Model([enc_in, dec_in], out)
model.compile(optimizer='adam', loss='categorical_crossentropy')

model.fit([X,y], y, epochs=2, verbose=0)

pred = model.predict([X[:1], y[:1]])

print(pred[0]) 
