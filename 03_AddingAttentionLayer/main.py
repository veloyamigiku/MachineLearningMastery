import argparse
import tensorflow as tf


#from utils import get_fib_seq
from utils import create_RNN
from utils import create_RNN_with_attention
from utils import get_fib_XY

def main(args):

  time_steps = 20
  hidden_units = 2
  epochs = 30
  
  trainX, trainY, testX, testY, scaler = get_fib_XY(
    total_fib_numbers=1200,
    time_steps=time_steps,
    train_percent=0.7
  )
  print('trainX = ', trainX)
  print('trainY = ', trainY)
  print('testX = ', testX)
  print('testY = ', testY)
  
  model_RNN = create_RNN(
    hidden_units=hidden_units,
    dense_units=1,
    input_shape=(time_steps, 1),
    activation=['tanh', 'tanh']
  )
  model_RNN.summary()

  trainX, trainY, testX, testY, scaler = get_fib_XY(
    total_fib_numbers=1200,
    time_steps=time_steps,
    train_percent=0.7)
  
  model_RNN.fit(
    x=trainX,
    y=trainY,
    epochs=epochs,
    batch_size=16,
    verbose=2)
  
  train_mse = model_RNN.evaluate(trainX, trainY)
  test_mse = model_RNN.evaluate(testX, testY)

  print("Train set MSE = ", train_mse)
  print("Test set MSE = ", test_mse)

  model_attention = create_RNN_with_attention(
    hidden_units=hidden_units,
    dense_units=1,
    input_shape=(time_steps, 1),
    activation='tanh')
  model_attention.summary()

  model_attention.fit(
    x=trainX,
    y=trainY,
    epochs=epochs,
    batch_size=16,
    verbose=2)
  
  train_mse_attn = model_attention.evaluate(trainX, trainY)
  test_mse_attn = model_attention.evaluate(testX, testY)

  print("Train set MSE(ATTN) = ", train_mse_attn)
  print("Test set MSE(ATTN) = ", test_mse_attn)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  args = parser.parse_args()
  main(args=args)
