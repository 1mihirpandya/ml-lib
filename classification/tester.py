import numpy as np
import matplotlib.pyplot as plt
import random
from BinaryClassifier import LogisticRegression
from MultiClassClassification import SoftmaxRegression
from sklearn.datasets import load_iris

x = []
labels = []
with open("data_banknote_authentication.txt", "r") as f:
    line = f.readline()
    while line:
        arr = [float(val) for val in line.strip().split(",")]
        arr[-1] = int(arr[-1])
        x.append(arr[:-1])
        labels.append(1 if arr[-1] == 1 else -1)
        line = f.readline()
x = np.array(x).T
labels = np.array(labels)

model = LogisticRegression()
model.init(x, labels)
model.run(batch_size=100, learning_rate=0.01, epochs=4000, show_loss=True)
pp = model.pred(x)
c = 0
for idx in range(len(pp)):
    if pp[idx] and labels[idx] == 1:
        c += 1
    elif not pp[idx] and labels[idx] == -1:
        c += 1
print(c/len(labels))
model.draw()

data = load_iris()
x = data.data.T
labels = data.target

model = SoftmaxRegression()
model.init(x, labels, 3)
model.run(batch_size=100, learning_rate=0.01, epochs=4000, show_loss=True)
pp = model.pred(x)
c = 0
for idx in range(len(pp)):
    if pp[idx] == labels[idx]:
        c += 1
print(c/len(labels))
model.draw()
plt.show()
