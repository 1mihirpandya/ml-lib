import numpy as np
import matplotlib.pyplot as plt
import random
from BinaryClassifier import LogisticRegression

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
"""
y = np.array(labels)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(x, y)
ys = clf.predict(x)
cc = 0
for v in range(len(ys)):
    if ys[v] == y[v]:
        cc += 1
print(cc/len(y))
print(clf.score(x,y))
"""
model = LogisticRegression()
model.init(x, labels)
model.run(learning_rate=0.001, epochs=5000, show_loss=True)
model.test()
model.draw()
plt.show()
