import matplotlib.pyplot as plt
import random
import numpy as np

class LinearRegression:
    def __init__(self, order=1, x=None, y=None):
        self.order = order + 1
        self.weights = np.zeros(self.order)
        self.raw_func = lambda x: np.array([x**exp for exp in range(self.order - 1, -1, -1)])
        self.loss = []
        if x and y:
            self.init(x, y)

    def init(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        std_x = np.std(self.x)
        mean_x = np.mean(self.x)
        self.x = (self.x - mean_x)/std_x
        std_y = np.std(self.y)
        mean_y = np.mean(self.y)
        self.y = (self.y - mean_y)/std_y
        ax2.scatter(self.x, self.y)

    def mse(self, pred, y):
        return np.sum((y - pred)**2)/len(y)

    def rmse(self, pred, y):
        return np.sqrt(self.mse(pred, y))

    def rmse_grad(self, pred, x, y):
        mse = self.mse(pred, y)
        return -1.0/len(y) * (mse ** -0.5) * np.dot(self.raw_func(x), (y - pred))

    def run(self, batch_size=10, learning_rate=0.001, epochs=1000):
        for epoch in range(epochs):
            idxs = np.random.choice(self.x.size, batch_size, replace=True)
            idxs.sort()
            mini_batch_x = np.array([self.x[idx] for idx in idxs])
            mini_batch_y = np.array([self.y[idx] for idx in idxs])
            preds = np.dot(self.weights, self.raw_func(mini_batch_x))

            #d_weights = self.rmse_grad(preds, mini_batch_x, mini_batch_y)#(-1/(2.0 * batch_size)) * np.dot(self.raw_func(mini_batch_x), 1/(mini_batch_y - preds))
            d_weights = (-2.0 / batch_size) * np.dot(self.raw_func(mini_batch_x), (mini_batch_y - preds))
            self.loss.append(self.rmse(preds, mini_batch_y))
            print(d_weights)
            print("d_weights^")
            self.weights = self.weights - learning_rate * d_weights
            print(self.weights)
            print("weights^\n")

    def show(self):
        x = np.array(sorted(list(set(self.x))))
        preds = np.dot(self.weights, self.raw_func(x))
        ax2.plot(x, preds, color='green')
        print(len(self.loss))
        ax3.plot(range(7000), self.loss)
        plt.show()

fig2, (ax2, ax3) = plt.subplots(nrows=2, ncols=1)
x_vals = x_vals = x = np.array(sorted([random.randint(-100, 100) for p in range(0, 1000)]))
y_vals = np.array([x**3 + -6.5 * x**2 + random.randint(-500,1000) for x in x_vals])
#y_vals = np.array([1.5 * x**2 + x + random.randint(-50,5) for x in x_vals])
data = zip(x_vals, y_vals)
model = LinearRegression(order=3)
model.init(x_vals, y_vals)
model.run(epochs=7000)
model.show()
#preds = [func(x) for x in x_vals]
#plt.plot(x_vals, preds, color='green')
#plt.show()
