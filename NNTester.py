import numpy as np
from NeuralNetwork import NeuralNetwork#, NeuralNetwork1, NeuralNetwork2
from sklearn.datasets import load_iris
from tensorflow.keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
#print(X_train)
X_train_transform = []
X_test_transform = []
for idx in range(len(X_train)):
    X_train_transform.append(X_train[idx].reshape(784))
for idx in range(len(X_test)):
    X_test_transform.append(X_test[idx].reshape(784))
X_train_transform = np.array(X_train_transform) / 255
X_test_transform = np.array(X_test_transform) / 255

avg_acc = 0
nn = NeuralNetwork()
nn.init(X_train_transform, Y_train, 10, 64)
nn.run(batch_size=512, learning_rate=0.001, epochs=5000)
c = 0
pp = nn.pred(X_test_transform) == Y_test
for val in pp:
    if val:
        c += 1
print(c/len(Y_test))
