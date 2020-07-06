import matplotlib.pyplot as plt
import random
import numpy as np

class LinearRegression:
    def __init__(self, order=1, x=None, y=None):
        self.order = order + 1
        self.weights = np.zeros(self.order)
        self.raw_func = lambda x: np.array([x**exp for exp in range(self.order - 1, -1, -1)])
        self.loss = lambda y_true, y_pred: (y_true - y_pred)**2
        if x and y:
            self.init(x, y)

    def init(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)

    def run(self, batch_size=10, learning_rate=0.0001, epochs=1000):
        for epoch in range(epochs):
            idxs = np.random.choice(self.x.size, batch_size, replace=True)
            mini_batch_x = np.array([self.x[idx] for idx in idxs])
            mini_batch_y = np.array([self.y[idx] for idx in idxs])
            print(mini_batch_x)
            print("mini_batch_x^")
            print(mini_batch_y)
            print("mini_batch_y^")
            print(self.raw_func(mini_batch_x))
            print("raw^")
            print(np.dot(self.weights, self.raw_func(mini_batch_x)))
            print("preds^")

            preds = np.dot(self.weights, self.raw_func(mini_batch_x))


            print((mini_batch_y - preds))
            print("piecewise^")
            print(np.dot(self.raw_func(mini_batch_x), (mini_batch_y - preds)))
            print("d_weights^")

            d_weights = (-2.0/batch_size) * np.dot(self.raw_func(mini_batch_x), (mini_batch_y - preds))
            self.weights = self.weights - learning_rate * d_weights
            print(self.weights)
            print("weights^\n")

    def show(self):
        preds = np.dot(self.weights, self.raw_func(self.x))
        plt.plot(self.x, preds, color='green')
        plt.show()


x_vals = np.array(range(-100,100))
y_vals = np.array([1.5 * x**2 + x + random.randint(-50,5) for x in x_vals])
data = zip(x_vals, y_vals)
plt.scatter(x_vals, y_vals)
model = LinearRegression(order=2)
model.init(x_vals, y_vals)
model.run(learning_rate=0.000001)
model.show()
#preds = [func(x) for x in x_vals]
#plt.plot(x_vals, preds, color='green')
#plt.show()

"""
m = 1.5
c = 0.1
func = lambda x: m * x + c
loss = lambda idx, pred: (y_vals[idx] - pred)**2
preds = [func(x) for x in x_vals]
plt.plot(x_vals, preds, color='red')
#plt.plot(x_vals, [loss(idx, preds[idx]) for idx in range(len(preds))], color='green')
m = 0
c = 0
cost_history = []
for epoch in range(2000):
    mini_batch = np.random.choice(x_vals.size, 10, replace=True)
    mini_batch_x = np.array([x_vals[idx] for idx in mini_batch])
    mini_batch_y = np.array([y_vals[idx] for idx in mini_batch])
    print(len(mini_batch_x), len(mini_batch_y))
    #print(mini_batch_x, mini_batch_y)
    preds = func(mini_batch_x)
    #print("mini_batch_x ", mini_batch_x)
    #print("mini_batch_y ", mini_batch_y)
    #print("preds ", preds)
    print()
    #print(np.sum(mini_batch_x * (mini_batch_y - preds)))
    #print(np.dot(mini_batch_x, (mini_batch_y - preds)))
    #print(np.dot(mini_batch_x, (mini_batch_y - preds)))
    d_m = -0.2 * np.sum(mini_batch_x * (mini_batch_y - preds))
    d_c = -0.2 * np.sum(mini_batch_y - preds)
    #print(d_m, d_c)
    m = m - 0.0001 * d_m
    c = c - 0.001 * d_c
    #print(m, c)
preds = [func(x) for x in x_vals]
plt.plot(x_vals, preds, color='green')
plt.show()
"""
