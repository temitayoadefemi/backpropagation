import numpy as np


np.random.seed(0)
input_layer, hidden_layer, output_layer = 2, 10, 1
w1 = np.random.normal(scale=0.1, size=(input_layer, hidden_layer))
b1 = np.zeros(hidden_layer)
w2 = np.random.normal(scale=0.1, size=(hidden_layer, output_layer))
b2 = np.zeros(output_layer)


X = [[0, 1], [1, 1], [1, 0], [0, 0]]
y = [[1], [0], [1], [0]]




