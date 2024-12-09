import argparse
from numpy import array
from numpy import random
from numpy import dot
from scipy.special import softmax

def main(args):
  
  # 4つの異なる単語の埋め込みを定義する。
  word_1 = array([1, 0, 0])
  word_2 = array([0, 1, 0])
  word_3 = array([1, 1, 0])
  word_4 = array([0, 0, 1])
  words = array([
    word_1,
    word_2,
    word_3,
    word_4
  ])

  # 重み(キュー、キー、バリュー)行列を作成する。
  random.seed(42)
  W_Q = random.randint(3, size=(3, 3))
  W_K = random.randint(3, size=(3, 3))
  W_V = random.randint(3, size=(3, 3))

  # 単語の埋め込みと重み(キュー、キー、バリュー)行列でベクトルを作成する。
  query_1 = word_1 @ W_Q
  key_1 = word_1 @ W_K
  value_1 = word_1 @W_V

  query_2 = word_2 @ W_Q
  key_2 = word_2 @ W_K
  value_2 = word_2 @W_V

  query_3 = word_3 @ W_Q
  key_3 = word_3 @ W_K
  value_3 = word_3 @W_V

  query_4 = word_4 @ W_Q
  key_4 = word_4 @ W_K
  value_4 = word_4 @W_V

  Q = words @ W_Q
  K = words @ W_K
  V = words @ W_V
  
  # 最初のクエリーとすべてのキーでスコア付けする。
  scores = array([
    dot(query_1, key_1),
    dot(query_1, key_2),
    dot(query_1, key_3),
    dot(query_1, key_4),
  ])

  scores = Q @ K.transpose()

  #weights = softmax(scores / key_1.shape[0] ** 0.5)
  weights = softmax(scores / K.shape[1] ** 0.5, axis=1)

  #attention = (weights[0] * value_1) + (weights[1] * value_2) + (weights[2] * value_3) + (weights[3] * value_4)
  attention = weights @ V
  print(attention)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  args = parser.parse_args()
  main(args=args)
