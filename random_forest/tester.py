from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from RandomForest import DecisionTree
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

dt = DecisionTree()
dt.init(data.data, data.target, ["cont" for _ in range(len(data.feature_names))])
cc = 0
for idx in range(len(data.target)):
    if dt.predict(data.data[idx]) == data.target[idx]:
        cc += 1
print(cc/len(data.target))
