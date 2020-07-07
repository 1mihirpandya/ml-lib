import matplotlib.pyplot as plt
import random
import numpy as np

class LogisticRegression:
    def __init__(self, x=None, y=None):
        self.dim = None
        self.weights = None
        self.loss = []
        self.f1 = plt.figure("Linear Regression")
        self.epochs = 0
        self.mean_x = []
        self.std_x = []
        self.labels = None
        if x and y:
            self.init(x, y)

    #for classification, multiple features will be used, so expecting a numpy array for x
    def init(self, x, y):
        self.x = x
        self.dim = self.x.shape
        self.weights = np.ones(self.dim[0])
        self.labels = np.array(y)
        for feature_idx in range(self.dim[0]):
            row = self.x[feature_idx]
            self.std_x.append(np.std(row))
            self.mean_x.append(np.mean(row))
            self.x[feature_idx] = (row - self.mean_x[-1])/self.std_x[-1]

    def log_loss(self):
        preds = np.dot(self.weights, self.x)
        #ce = -np.sum(self.labels * np.log(preds)) / self.dim[1]
        #print(ce, np.sum(np.log(1 + np.exp(-1 * self.labels * preds))))
        return np.sum(np.log(1 + np.exp(-1 * self.labels * preds)))

    def log_loss_grad(self, pred, x, labels):
        piece = np.exp(-1 * labels * pred)
        numerator = np.dot(-1 * labels, piece)
        denominator = 1 + np.sum(piece)
        return np.sum((numerator / denominator) * x)

    def run(self, batch_size=10, learning_rate=0.0001, epochs=1000, show_loss=False):
        self.epochs = epochs
        for epoch in range(epochs):
            idxs = np.random.choice(self.dim[1], batch_size, replace=True)
            idxs.sort()
            mini_batch_x = np.array([self.x[:, idx] for idx in idxs]).T
            mini_batch_labels = np.array([self.labels[idx] for idx in idxs])
            preds = np.dot(self.weights, mini_batch_x)
            d_weights = self.log_loss_grad(preds, mini_batch_x, mini_batch_labels)
            self.weights = self.weights - learning_rate * d_weights
            if show_loss:
                self.loss.append(self.log_loss())

    def test(self):
        count = 0
        preds = np.dot(self.weights, self.x)
        preds = [(1 if pred >= 0 else -1) for pred in preds]
        for idx in range(len(preds)):
            if preds[idx] == self.labels[idx]:
                count += 1
        print(count/self.dim[1])



    def draw(self):
        if len(self.loss) == self.epochs:
            self.ax1 = self.f1.add_subplot(111)
            #print(self.loss)
            self.ax1.plot(range(self.epochs), self.loss)
        plt.draw()
