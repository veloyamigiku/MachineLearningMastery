import argparse
from keras.api.models import load_model
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.layers import SimpleRNN
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

def create_RNN(
  hidden_units: int,
  dense_units: int,
  input_shape,
  activation):
  
  model = Sequential()
  model.add(SimpleRNN(
    hidden_units,
    input_shape=input_shape,
    activation=activation[0]
  ))
  model.add(Dense(
    units=dense_units,
    activation=activation[1]))
  model.compile(
    loss='mean_squared_error',
    optimizer='adam')
  
  return model

def get_train_test(
  url: str,
  split_percent: int = 0.8):
  df = pd.read_csv(
    url,
    usecols=[1])
  data = np.array(df.values.astype('float32'))
  scaler = MinMaxScaler(feature_range=(0, 1))
  data = scaler.fit_transform(data).flatten()
  n = len(data)
  split = int(n * split_percent)
  train_data = data[range(split)]
  test_data = data[split:]
  return train_data, test_data, data

def get_XY(
  dat,
  time_steps: int):
  Y_ind = np.arange(time_steps, len(dat), time_steps)
  Y = dat[Y_ind]
  rows_x = len(Y)
  X = dat[range(time_steps * rows_x)]
  X = np.reshape(X, (rows_x, time_steps, 1))
  return X, Y

def print_error(
  trainY,
  testY,
  train_predict,
  test_predict):
  train_rmse = np.sqrt(mean_squared_error(trainY, train_predict))
  test_rmse = np.sqrt(mean_squared_error(testY, test_predict))
  print('Train RMSE: %.3f RMSE' % (train_rmse))
  print('Test RMSE: %.3f RMSE' % (test_rmse))

def plot_result(
  trainY,
  testY,
  train_predict,
  test_predict,
  output_path):
  actual = np.append(trainY, testY)
  predictions = np.append(train_predict, test_predict)
  rows = len(actual)
  plt.figure(figsize=(15, 6), dpi=80)
  plt.plot(range(rows), actual)
  plt.plot(range(rows), predictions)
  plt.axvline(x=len(trainY), color='r')
  plt.legend(['Actual', 'Predictions'])
  plt.xlabel('Observation number after given time steps')
  plt.ylabel('Sunspots scaled')
  plt.title('Actual and Predicted Values. The Red Line Separates The Training And Test Examples')
  plt.savefig(output_path)
  plt.close()
  
def main(args):
  """
  demo_model = create_RNN(
    hidden_units=2,
    dense_units=1,
    input_shape=(3, 1),
    activation=['linear', 'linear']
  )
  print(demo_model.summary())

  wx = demo_model.get_weights()[0]
  wh = demo_model.get_weights()[1]
  bh= demo_model.get_weights()[2]
  wy = demo_model.get_weights()[3]
  by = demo_model.get_weights()[4]
  print('wx = ', wx, ' wh = ', wh, ' bh = ', bh, ' wy = ', wy, ' by = ', by)

  x = np.array([1, 2, 3])
  # 入力を(サンプルサイズ,タイムステップ数,特徴量の数)に整形する。
  x_input = np.reshape(x, (1, 3, 1))
  y_pred_model = demo_model.predict(x_input)
  m = 2
  h0 = np.zeros(m)
  h1 = np.dot(x[0], wx) + np.dot(h0, wh) + bh
  h2 = np.dot(x[1], wx) + np.dot(h1, wh) + bh
  h3 = np.dot(x[2], wx) + np.dot(h2, wh) + bh
  o3 = np.dot(h3, wy) + by
  print('h1 = ', h1, ' h2 = ', h2, ' h3 = ', h3)
  print('Prediction from network ', y_pred_model)
  print('Prediction from our computation ', o3)
  """

  # Reading Data and Splitting Into Train and Test
  dataset_path = args.output_dir + os.sep + 'dataset.pkl'
  if os.path.exists(dataset_path):
    with open(dataset_path, mode='rb') as f:
      dataset = pickle.load(f)
    train_data = dataset['train_data']
    test_data = dataset['test_data']
    data = dataset['data']
  else:
    sunspots_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv'
    train_data, test_data, data = get_train_test(sunspots_url)
    with open(dataset_path, mode='wb') as fw:
      pickle.dump(
        {
          'train_data': train_data,
          'test_data': test_data,
          'data': data
        },
        fw
      )
  
  # Reshaping Data for Keras
  time_steps = 12
  trainX, trainY = get_XY(
    dat=train_data,
    time_steps=time_steps)
  testX, testY = get_XY(
    dat=test_data,
    time_steps=time_steps)
  
  # Create RNN Model and Train
  model_path = args.output_dir + os.sep + 'model.keras'
  if os.path.exists(model_path):
    model = load_model(model_path)
  else:
    model = create_RNN(
      hidden_units=3,
      dense_units=1,
      input_shape=(time_steps, 1),
      activation=['tanh', 'tanh']
    )
    model.fit(
      x=trainX,
      y=trainY,
      epochs=20,
      batch_size=1,
      verbose=2
    )
    model.save(args.output_dir + os.sep + 'model.keras')
  
  # Compute and Print the Root Mean Square Error
  train_predict = model.predict(trainX)
  test_predict = model.predict(testX)
  print_error(
    trainY=trainY,
    testY=testY,
    train_predict=train_predict,
    test_predict=test_predict)
  
  # View the Result
  plot_result(
    trainY=trainY,
    testY=testY,
    train_predict=train_predict,
    test_predict=test_predict,
    output_path=args.output_dir + os.sep + 'result.png'
  )

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_dir', type=str, default='.')
  args = parser.parse_args()
  main(args=args)
