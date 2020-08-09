from sklearn.datasets import load_breast_cancer, load_iris
from RandomForest import RandomForest
from DecisionTree import DecisionTree
import pandas as pd
import time

data = load_breast_cancer()

#Requires data to be in Pandas dataframe with targets (targets need to be the last column)
#All categorical variables need to be numerically classified (0, 1, 2, etc)

#Decision tree test
dt = DecisionTree()
d = pd.DataFrame(data.data)
d[len(d.columns)] = data.target
dt.init(d, ["cont" for _ in range(len(d.columns) - 1)])
cc = 0
for idx in range(len(data.data)):
    if dt.predict(data.data[idx]) == data.target[idx]:
        cc += 1
print(cc/len(data.target))

#Random forest test
dt = RandomForest()
d = pd.DataFrame(data.data)
d[len(d.columns)] = data.target
dt.init(d, ["cont" for _ in range(len(d.columns) - 1)], 5)
cc = 0
for idx in range(len(data.data)):
    if dt.predict(data.data[idx]) == data.target[idx]:
        cc += 1
print(cc/len(data.target))
