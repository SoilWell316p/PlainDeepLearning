import numpy as np

from typing import List

from DL_soilwell.train import train
from DL_soilwell.layers import Linear, Tanh
from DL_soilwell.nn import NeuralNet
from DL_soilwell.optim import SGD

def fizz_buzz_encoding(x: int) -> List[int]:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]

def binary_encoding(x: int) -> List[int]:
    # encode the inputs into 10 digits binary
    # so that NN can learn
    return [x >> i & 1 for i in range(10)]

inputs = np.array([
    binary_encoding(x)
    for x in range(101, 1024)
])

targets = np.array([
    fizz_buzz_encoding(x)
    for x in range(101, 1024)
])

net = NeuralNet([
    Linear(input_size=10, output_size=50),
    Tanh(),
    Linear(input_size=50, output_size=4),
])

train(net,
      inputs,
      targets,
      num_epochs=5000,
      optimizer=SGD(lr=0.001))

count = 0
for x in range(1, 101):
    predicted = net.forward(binary_encoding(x))
    predicted_idx = np.argmax(predicted)
    actual_idx = np.argmax(fizz_buzz_encoding(x))
    labels = [str(x), "fizz", "buzz", "fizzbuzz"]
    print(x, labels[predicted_idx], labels[actual_idx])
    if predicted_idx == actual_idx:
        count += 1

print("accuracy: {:.2f}".format(count/101))



