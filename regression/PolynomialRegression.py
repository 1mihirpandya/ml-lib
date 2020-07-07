import matplotlib.pyplot as plt
import random
import numpy as np

class PolynomialRegression:
    def __init__(self, order=1, x=None, y=None):
        self.order = order + 1
        self.weights = np.zeros(self.order)
        self.raw_func = lambda x: np.array([x**exp for exp in range(self.order - 1, -1, -1)])
        self.loss = []
        f1 = plt.figure("Polynomial Regression")
        self.ax1 = f1.add_subplot(111)
        self.epochs = 0
        self.mean_x = 0
        self.mean_y = 0
        self.std_x = 0
        self.std_y = 0
        self.unscaled_x = None
        self.unscaled_y = None
        if x and y:
            self.init(x, y)

    def init(self, x, y):
        self.unscaled_x = np.array(x)
        self.unscaled_y = np.array(y)
        self.ax1.scatter(self.unscaled_x, self.unscaled_y)
        self.std_x = np.std(self.unscaled_x)
        self.mean_x = np.mean(self.unscaled_x)
        self.x = (self.unscaled_x - self.mean_x)/self.std_x
        self.std_y = np.std(self.unscaled_y)
        self.mean_y = np.mean(self.unscaled_y)
        self.y = (self.unscaled_y - self.mean_y)/self.std_y

    def batch_mse(self, pred, y):
        return np.sum((y - pred)**2)/len(y)

    def mse(self):
        preds = np.dot(self.weights, self.raw_func(self.x))
        return np.sum((self.y - preds)**2)/len(self.y)

    def rmse(self):
        return np.sqrt(self.mse())

    def rmse_grad(self, pred, x, y):
        mse = self.batch_mse(pred, y)
        return -1.0/len(y) * (mse ** -0.5) * np.dot(self.raw_func(x), (y - pred))

    def run(self, batch_size=10, learning_rate=0.0001, epochs=1000, show_loss=False):
        self.epochs = epochs
        for epoch in range(epochs):
            idxs = np.random.choice(self.x.size, batch_size, replace=True)
            idxs.sort()
            mini_batch_x = np.array([self.x[idx] for idx in idxs])
            mini_batch_y = np.array([self.y[idx] for idx in idxs])
            preds = np.dot(self.weights, self.raw_func(mini_batch_x))
            d_weights = self.rmse_grad(preds, mini_batch_x, mini_batch_y)
            #d_weights = (-2.0 / batch_size) * np.dot(self.raw_func(mini_batch_x), (mini_batch_y - preds))
            self.weights = self.weights - learning_rate * d_weights
            if show_loss:
                self.loss.append(self.rmse())

    def draw(self):
        preds = (np.dot(self.weights, self.raw_func(self.x)) * self.std_y) + self.mean_y
        self.ax1.plot(self.unscaled_x, preds, color='red')
        if len(self.loss) == self.epochs:
            f2 = plt.figure("Polynomial Regression Loss")
            ax2 = f2.add_subplot(111)
            ax2.plot(range(self.epochs), self.loss)
        plt.draw()
