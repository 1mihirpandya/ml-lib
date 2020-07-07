import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import random
from LinearRegression import LinearRegression
from PolynomialRegression import PolynomialRegression

#Linear test
x_vals = np.array([random.randint(-100, 100) for p in range(0, 1000)])
x_vals.sort()
y_vals = np.array([1.5 * x + random.randint(-50,100) for x in x_vals])
model = LinearRegression()
model.init(x_vals, y_vals)
model.run(learning_rate=0.001, epochs=9000, show_loss=True)
model.draw()

#Polynomial test
y_vals2 = np.array([x**2 + -6.5 * x + random.randint(-500,1000) for x in x_vals])
model = PolynomialRegression(order=2)
model.init(x_vals, y_vals2)
model.run(learning_rate=0.001, epochs=7000,show_loss=True)
model.draw()

plt.show()
