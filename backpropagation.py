import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

np.random.seed(0)
input_layer, hidden_layer, output_layer = 2, 10, 1
w1 = np.random.normal(scale=0.1, size=(input_layer, hidden_layer))
b1 = np.zeros(hidden_layer)
w2 = np.random.normal(scale=0.1, size=(hidden_layer, output_layer))
b2 = np.zeros(output_layer)


X = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])
y = np.array([[1], [0], [1], [0]])


learning_rate = 0.5
epochs = 10000

for epoch in range(epochs):
    for x, target in zip(X, y):
        x = x.reshape(-1, 1)
        hidden_input = np.dot(w1.T, x) + b1.reshape(-1, 1)
        hidden_output = sigmoid(hidden_input)
        final_input = np.dot(w2.T, hidden_output) + b2.reshape(-1, 1)
        final_output = sigmoid(final_input)
        error = target - final_output
        output_delta = error * sigmoid_derivative(final_output)
        hidden_error = w2.dot(output_delta.T)
        hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

        w2 += learning_rate * hidden_output.dot(output_delta.T)
        b2 += learning_rate * output_delta.flatten()
        w1 += learning_rate * x.dot(hidden_delta.T)
        b1 += learning_rate * hidden_delta.flatten()


def infer(x, w1, w2, b1, b2):
    x = np.array(x).reshape(-1, 1)
    # Input to Hidden Layer
    hidden_input = np.dot(w1.T, x) + b1.reshape(-1, 1)
    hidden_output = sigmoid(hidden_input)
    
    # Hidden Layer to Output Layer
    final_input = np.dot(w2.T, hidden_output) + b2.reshape(-1, 1)
    final_output = sigmoid(final_input)

    return final_output

result = infer([0, 0], w1=w1, w2=w2, b1=b1, b2=b2)
print(result)




