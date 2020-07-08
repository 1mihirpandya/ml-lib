import matplotlib.pyplot as plt
import random
import numpy as np

class LogisticRegression:
    def __init__(self, x=None, y=None):
        self.dim = None
        self.weights = None
        self.loss = []
        self.f1 = plt.figure("Logistic Regression Loss")
        self.epochs = 0
        self.mean_x = []
        self.std_x = []
        self.labels = None
        if x and y:
            self.init(x, y)

    #for classification, multiple features will be used, so expecting a numpy array for x
    def init(self, x, y):
        self.x = x.copy()
        self.dim = self.x.shape
        self.weights = np.zeros(self.dim[0] + 1) + 0.5
        self.labels = np.array(y)
        for feature_idx in range(self.dim[0]):
            row = self.x[feature_idx]
            self.std_x.append(np.std(row))
            self.mean_x.append(np.mean(row))
            self.x[feature_idx] = (row - self.mean_x[-1])/self.std_x[-1]
        added = np.ones((1,self.dim[1]))
        self.x = np.append(self.x, added, axis=0)

    def run(self, batch_size=10, learning_rate=0.0001, epochs=1000, show_loss=False):
        self.epochs = epochs
        for epoch in range(epochs):
            idxs = np.random.choice(self.dim[1], batch_size, replace=True)
            idxs.sort()
            mini_batch_x = np.array([self.x[:, idx] for idx in idxs]).T
            mini_batch_labels = np.array([self.labels[idx] for idx in idxs])
            h = 1 / (1 + np.exp(-np.dot(self.weights, mini_batch_x)))
            gradient = np.dot(mini_batch_x, (h - mini_batch_labels)) / mini_batch_labels.size
            self.weights = self.weights - learning_rate * gradient
            if show_loss:
                self.loss.append(self.log_loss())

    def pred(self, x):
        x_copy = x.copy()
        for feature_idx in range(self.dim[0]):
            row = x_copy[feature_idx]
            x_copy[feature_idx] = (row - self.mean_x[feature_idx])/self.std_x[feature_idx]
        added = np.ones((1,x_copy.shape[1]))
        x_copy = np.append(x_copy, added, axis=0)
        preds = self.sigmoid(np.dot(self.weights, x_copy))
        return preds >= 0.5

    def draw(self):
        if len(self.loss) == self.epochs:
            self.ax1 = self.f1.add_subplot(111)
            #print(self.loss)
            self.ax1.plot(range(self.epochs), self.loss)
        plt.draw()
