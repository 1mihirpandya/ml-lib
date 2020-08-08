from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from RandomForest import DecisionTree
import pandas as pd
import time

from sklearn import tree
data = load_breast_cancer()
iris = load_iris()
print()
clf = tree.DecisionTreeClassifier()
clf.fit(data.data, data.target)
preds = clf.predict(data.data)
cc = 0
for idx in range(len(preds)):
    if preds[idx] == data.target[idx]:
        cc += 1
print(cc/len(data.target))

print(time.time())
dt = DecisionTree()
d = pd.DataFrame(data.data)
d[len(d.columns)] = data.target
dt.init(d, ["cont" for _ in range(len(data.feature_names))])
print(time.time())
cc = 0
for idx in range(len(data.target)):
    if dt.predict(data.data[idx]) == data.target[idx]:
        cc += 1
print(cc/len(data.target))
