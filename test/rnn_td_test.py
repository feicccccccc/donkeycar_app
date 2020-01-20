# A exercise to understand how to implement sequence model in keras

import numpy as np

import tensorflow as tf

from tensorflow.python import keras
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import Dense

length = 5
seq = np.array([i / float(length) for i in range(length)])

print(seq)

X = seq.reshape(1, 5, 1)
Y = seq.reshape(1, 5)

# Input shape: (batch, time_step, feature)
# Output shape: (batch, pred)

n_neuron = 3
n_batch = length
n_epoch = 1000

model = Sequential()
model.add(LSTM(n_neuron, input_shape=(1, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer="Adam")

print(model.summary())
# number of parameters:
#  n = 4 * ((inputs + 1) * outputs + outputs^2)
# 4 gate: forget gate, update gate & what to update, output gate.
# each gate have 1. Hidden unit part (hidden * hidden), 2. input part (hidden * output)
# +1 for bias

model.fit(X, Y, epochs=n_epoch, batch_size=n_batch, verbose=2)
result = model.predict(X, batch_size=n_batch, verbose=0)

for value in result:
    print('%.1f' % value)
