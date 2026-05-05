import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

text = "hello"
chars = list(set(text))
c2i = {c:i for i,c in enumerate(chars)}
i2c = {i:c for c,i in c2i.items()}

# Data
X, y = [], []
for i in range(len(text)-1):
    x = [0]*len(chars); x[c2i[text[i]]] = 1
    y1 = [0]*len(chars); y1[c2i[text[i+1]]] = 1
    X.append(x); y.append(y1)

X = np.array(X).reshape(len(X),1,len(chars))
y = np.array(y)

# Model
model = Sequential([
    LSTM(8, input_shape=(1,len(chars))),
    Dense(len(chars), activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train
model.fit(X, y, epochs=100, verbose=0)

# Generate
inp = X[0]
out = ""
for _ in range(5):
    p = model.predict(inp.reshape(1,1,len(chars)), verbose=0)
    idx = np.argmax(p)
    out += i2c[idx]
    inp = np.eye(len(chars))[idx]

print(out)
