import matplotlib.pyplot as plt
import random
import numpy as np

class LinearRegression:
    def __init__(self, x=None, y=None):
        self.m = 0
        self.c = 0
        self.loss = []
        f1 = plt.figure("Linear Regression")
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

    def mse(self):
        preds = (self.m * self.x) + self.c
        return np.sum((self.y - preds)**2)/len(self.y)

    def run(self, batch_size=10, learning_rate=0.0001, epochs=1000, show_loss=False):
        self.epochs = epochs
        for epoch in range(epochs):
            idxs = np.random.choice(self.x.size, batch_size, replace=True)
            idxs.sort()
            mini_batch_x = np.array([self.x[idx] for idx in idxs])
            mini_batch_y = np.array([self.y[idx] for idx in idxs])
            preds = (self.m * mini_batch_x) + self.c
            d_m = (-2.0 / batch_size) * np.dot(mini_batch_x, (mini_batch_y - preds))
            d_c = (-2.0 / batch_size) * np.sum((mini_batch_y - preds))
            self.m -= learning_rate * d_m
            self.c -= learning_rate * d_c
            if show_loss:
                self.loss.append(self.mse())

    def draw(self):
        preds = ((self.m * self.x + self.c) * self.std_y) + self.mean_y
        self.ax1.plot(self.unscaled_x, preds, color='red')
        if len(self.loss) == self.epochs:
            f2 = plt.figure("Linear Regression Loss")
            ax2 = f2.add_subplot(111)
            ax2.plot(range(self.epochs), self.loss)
        plt.draw()
