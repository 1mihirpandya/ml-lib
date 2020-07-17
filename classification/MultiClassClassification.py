import matplotlib.pyplot as plt
import random
import numpy as np

class SoftmaxRegression:
    def __init__(self, x=None, y=None):
        self.dim = None
        self.weights = None
        self.loss = []
        self.f1 = plt.figure("Softmax Regression Loss")
        self.epochs = 0
        self.mean_x = []
        self.std_x = []
        self.labels = None
        self.classes = 0
        if x and y:
            self.init(x, y)

    #for classification, multiple features will be used, so expecting a numpy array for x
    def init(self, x, y, classes):
        self.classes = classes
        self.x = x.copy()
        self.dim = self.x.shape
        self.weights = np.array([np.zeros(self.dim[0] + 1) + 0.5 for num in range(self.classes)])
        self.labels = np.array(y)
        for feature_idx in range(self.dim[0]):
            row = self.x[feature_idx]
            self.std_x.append(np.std(row))
            self.mean_x.append(np.mean(row))
            self.x[feature_idx] = (row - self.mean_x[-1])/self.std_x[-1]
        added = np.ones((1,self.dim[1]))
        self.x = np.append(self.x, added, axis=0)

    def softmax(self):
        all_preds = np.exp(np.dot(self.weights, self.x))
        preds = np.array([all_preds[:,idx][int(self.labels[idx])] for idx in range(len(self.labels))])
        all_preds = np.sum(all_preds, axis=0)
        med = (preds/all_preds)
        dist = -np.log(med)
        return np.sum(dist)/len(self.labels)

    def run(self, batch_size=10, learning_rate=0.0001, epochs=1000, show_loss=False):
        self.epochs = epochs
        for epoch in range(epochs):
            idxs = np.random.choice(self.dim[1], batch_size, replace=False)
            idxs.sort()
            mini_batch_x = np.array([self.x[:, idx] for idx in idxs]).T
            mini_batch_labels = np.array([self.labels[idx] for idx in idxs])

            all_preds = np.exp(np.dot(self.weights, mini_batch_x))
            preds = np.array([np.array([all_preds[:,idx][i] for idx in range(len(mini_batch_labels))]) for i in range(self.classes)])
            k_vals = np.array([np.array([(1 if mini_batch_labels[idx] == i else 0) for idx in range(len(mini_batch_labels))]) for i in range(self.classes)])
            all_preds = np.sum(all_preds, axis=0)
            gradient = -np.dot(k_vals - (preds/all_preds), mini_batch_x.T)
            self.weights = self.weights - learning_rate * gradient

            if show_loss:
                self.loss.append(self.softmax())

    def pred(self, x):
        x_copy = x.copy()
        for feature_idx in range(self.dim[0]):
            row = x_copy[feature_idx]
            x_copy[feature_idx] = (row - self.mean_x[feature_idx])/self.std_x[feature_idx]
        added = np.ones((1,x_copy.shape[1]))
        x_copy = np.append(x_copy, added, axis=0)

        all_preds = np.exp(np.dot(self.weights, x_copy))
        all_preds_sum = np.sum(all_preds, axis=0)
        preds = [(all_preds[:,idx]/np.sum(all_preds[:,idx])).argmax() for idx in range(self.dim[1])]
        return preds

    def draw(self):
        if len(self.loss) == self.epochs:
            self.ax1 = self.f1.add_subplot(111)
            self.ax1.plot(range(self.epochs), self.loss)
        plt.draw()
