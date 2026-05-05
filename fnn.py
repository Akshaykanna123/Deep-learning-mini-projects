import numpy as np
from keras.models import Sequential
from keras.layers import Dense

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

model = Sequential()
model.add(Dense(2, input_dim=2, activation='relu'))  # hidden layer
model.add(Dense(1, activation='sigmoid'))            # output layer

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=500, verbose=0)

print(model.predict(X))
