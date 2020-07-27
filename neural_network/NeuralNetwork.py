import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.weights = []
        self.bias = []
        self.loss = []
        self.epochs = 0
        self.x = None
        self.y = None

    def init(self, x, y, num_outputs, *layers):
        self.x = x.copy()
        self.y = np.array([np.zeros(num_outputs) for i in range(len(y))])
        for idx in range(len(y)):
            self.y[idx][y[idx]] = 1
        for idx in range(len(layers)):
            if idx == 0:
                self.weights.append(np.random.randn(self.x.shape[1], layers[idx]).astype("float64") * np.sqrt(1 / layers[idx]))
            else:
                self.weights.append(np.random.randn(self.weights[-1].shape[1], layers[idx]).astype("float64") * np.sqrt(1 / layers[idx]))
            self.bias.append(np.random.randn(layers[idx]).astype("float64") * np.sqrt(1. / layers[idx]))
        self.weights.append(np.random.randn(self.weights[-1].shape[1], num_outputs).astype("float64") * np.sqrt(1 / num_outputs))
        self.bias.append(np.random.randn(num_outputs).astype("float64") * np.sqrt(1. / num_outputs))

    def run(self, batch_size=10, learning_rate=0.0001, epochs=1000, show_loss=False):
        self.epochs = epochs
        for epoch in range(epochs):
            if epoch % 100 == 0:
                print("Epoch: {}, Loss: {}".format(epoch, self.cross_entropy_loss(self.feedforward(self.x)[-1], self.y)))
            idxs = np.random.choice(self.x.shape[0], batch_size, replace=False)
            idxs.sort()
            mini_batch_x = np.array([self.x[idx, :] for idx in idxs])
            mini_batch_y = np.array([self.y[idx] for idx in idxs])
            layer_outputs = self.feedforward(mini_batch_x)
            self.backprop(mini_batch_x, mini_batch_y, layer_outputs, learning_rate)

    def softmax(self, vals):
        ret = []
        for idx in range(len(vals)):
            x = vals[idx] - vals[idx].max()
            ret.append(np.exp(x) / np.sum(np.exp(x)))
        return np.array(ret)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def cross_entropy_loss(self, x, y):
        return -np.sum(np.sum(y * np.log(x), axis=1)/len(y))

    def sigmoid_grad(self, layer):
        return layer * (1 - layer)

    def softmax_grad(self, layer, y):
        return layer - y

    def feedforward(self, x):
        curr_layers = [x]
        for idx in range(len(self.weights)):
            val = np.matmul(curr_layers[-1], self.weights[idx]) + self.bias[idx]
            if idx == len(self.weights) - 1:
                curr_layers.append(self.softmax(val))
            else:
                curr_layers.append(self.sigmoid(val))
        return curr_layers

    def backprop(self, x, y, layer_outputs, learning_rate):
        layer_idx = len(layer_outputs) - 1
        deriv = (layer_outputs[layer_idx] - y)
        layer_idx -= 1
        for idx in range(len(self.weights) - 1, -1, -1):
            gradient = np.dot(layer_outputs[layer_idx].T, deriv)
            self.bias[idx] -= learning_rate * np.sum(deriv, axis=0)
            deriv = np.dot(deriv, self.weights[idx].T) * self.sigmoid_grad(layer_outputs[layer_idx])
            self.weights[idx] -= learning_rate * gradient
            layer_idx -= 1

    def pred(self, x):
        preds = self.feedforward(x.copy())[-1]
        return np.array([pred.argmax() for pred in preds])
