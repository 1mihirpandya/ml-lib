import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.weights = []
        self.loss = []
        #self.f1 = plt.figure("Loss")
        self.epochs = 0
        #self.mean_x = []
        #self.std_x = []
        self.x = None
        self.y = None

    def init(self, x, y, num_outputs, *layers):
        self.x = x.copy()
        self.y = y.copy()
        for idx in range(len(layers)):
            if idx == 0:
                self.weights.append(np.random.rand(self.x.shape[1], layers[idx]))
            else:
                self.weights.append(np.random.rand(self.weights[-1].shape[1], layers[idx]))
        self.weights.append(np.random.rand(self.weights[-1].shape[1], num_outputs))
        #print(self.weights)
        #self.weights = np.array(self.weights)
        print(self.weights)

    def run(self, batch_size=10, learning_rate=0.0001, epochs=1000, show_loss=False):
        self.epochs = epochs
        for epoch in range(epochs):
            idxs = np.random.choice(self.x.shape[0], batch_size, replace=False)
            idxs.sort()
            mini_batch_x = np.array([self.x[idx, :] for idx in idxs])
            #print(mini_batch_x)
            mini_batch_y = np.array([self.y[idx] for idx in idxs])
            self.loss.append(self.mse())
            #print(mini_batch_y)
            layer_outputs = self.feedforward(mini_batch_x)
            self.backprop(mini_batch_x, mini_batch_y, layer_outputs, learning_rate)

    def mse(self):
        return np.sum((1/len(self.y)) * (self.y - self.feedforward(self.x)[-1])**2)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def mse_grad(self, x, y):
        self.y - self.weights

    def sigmoid_grad(self, layer):
        return layer * (1 - layer)

    def feedforward(self, x):
        curr_layers = [x]
        for weight in self.weights:
            curr_layers.append(self.sigmoid(np.dot(curr_layers[-1], weight)))
        return curr_layers

    def backprop(self, x, y, layer_outputs, learning_rate):
        layer_idx = len(layer_outputs) - 1
        deriv = (-2.0 / len(y)) * self.sigmoid_grad(layer_outputs[layer_idx]) * (y - layer_outputs[layer_idx])
        layer_idx -= 1
        for idx in range(len(self.weights) - 1, -1, -1):
            gradient = np.dot(layer_outputs[layer_idx].T, deriv)
            deriv = np.dot(deriv, self.weights[idx].T) * self.sigmoid_grad(layer_outputs[layer_idx])
            self.weights[idx] -= learning_rate * gradient
            layer_idx -= 1

    def pred(self, x):
        copy_x = x.copy()
        curr_layer = self.sigmoid(np.dot(copy_x, self.weights[0]))
        for weight in self.weights[1:]:
            curr_layer = self.sigmoid(np.dot(curr_layer, weight))
        return curr_layer
