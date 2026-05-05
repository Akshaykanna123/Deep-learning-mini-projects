import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder

data = load_iris()
X = data.data
y = data.target.reshape(-1,1)

enc = OneHotEncoder(sparse_output=False)
y = enc.fit_transform(y)

np.random.seed(0)

model = Sequential()
model.add(Dense(5, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=100, verbose=0)

preds = model.predict(X[:5])

print(np.round(preds))
