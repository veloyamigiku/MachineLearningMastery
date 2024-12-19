import keras._tf_keras.keras.backend as K
from keras._tf_keras.keras.layers import Dense
from keras._tf_keras.keras.layers import Input
from keras._tf_keras.keras.layers import Layer
from keras._tf_keras.keras.layers import SimpleRNN
from keras._tf_keras.keras.metrics import mse
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.models import Sequential
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def get_fib_seq(
  n: int,
  scale_data: bool = True):
  
  seq = np.zeros(n)
  fib_n1 = 0.0
  fib_n = 1.0
  for i in range(n):
    seq[i] = fib_n1 + fib_n
    fib_n1 = fib_n
    fib_n = seq[i]
  scaler = None
  if scale_data:
    scaler = MinMaxScaler(feature_range=(0, 1))
    seq = np.reshape(seq, (n, 1))
    seq = scaler.fit_transform(seq).flatten()
  return seq, scaler

def get_fib_XY(
  total_fib_numbers: int,
  time_steps: int,
  train_percent: float,
  scale_data=True):

  dat, scaler = get_fib_seq(
    n=total_fib_numbers,
    scale_data=scale_data)
  Y_ind = np.arange(
    time_steps,
    len(dat),
    1)
  Y = dat[Y_ind]
  rows_x = len(Y)
  X = dat[0:rows_x]
  for i in range(time_steps - 1):
    temp = dat[i + 1:rows_x + i + 1]
    X = np.column_stack((X, temp))
  rand = np.random.RandomState(seed=13)
  idx = rand.permutation(rows_x)
  split = int(train_percent * rows_x)
  train_ind = idx[0:split]
  test_ind = idx[split:]
  trainX = X[train_ind]
  trainY = Y[train_ind]
  testX = X[test_ind]
  testY = Y[test_ind]
  trainX = np.reshape(trainX, (len(trainX), time_steps, 1))
  testX = np.reshape(testX, (len(testX), time_steps, 1))
  return trainX, trainY, testX, testY, scaler

def create_RNN(
  hidden_units: int,
  dense_units: int,
  input_shape: set,
  activation: list):

  model = Sequential()
  model.add(SimpleRNN(
    units=hidden_units,
    input_shape=input_shape,
    activation=activation[0]))
  model.add(Dense(
    units=dense_units,
    activation=activation[1]))
  model.compile(loss='mse', optimizer='adam')
  return model

class attention(Layer):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
  # input_shape=(None,time_steps,hidden_units)
  def build(self, input_shape):
    self.W = self.add_weight(
      name='attention_weight',
      shape=(input_shape[-1], 1),
      initializer='random_normal',
      trainable=True)
    self.b = self.add_weight(
      name='attention_bias',
      shape=(input_shape[1], 1),
      initializer='zeros',
      trainable=True)
    return super().build(input_shape)
  
  def call(self, x):
    e = K.tanh(K.dot(x, self.W) + self.b)
    #e = K.squeeze(e, axis=-1)
    alpha = K.softmax(e, axis=1)
    #alpha = K.expand_dims(alpha, axis=-1)
    context = x * alpha
    context = K.sum(context, axis=1)
    return context

def create_RNN_with_attention(
  hidden_units: int,
  dense_units: int,
  input_shape: set,
  activation: str):

  x = Input(shape=input_shape)
  RNN_layer = SimpleRNN(
    units=hidden_units,
    return_sequences=True,
    activation=activation)(x)
  attention_layer = attention()(RNN_layer)
  outputs = Dense(
    units=dense_units,
    trainable=True,
    activation=activation)(attention_layer)
  model = Model(x, outputs)
  model.compile(loss='mse', optimizer='adam')#, run_eagerly=True)
  return model
