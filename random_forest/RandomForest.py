import numpy as np
import pandas as pd
import random
import math

class DecisionTree:
    def __init__(self):
        self.left, self.right = None, None
        self.features = []

    def init(self, x, y, features):
        self.features = features #cat=categorical or cont=continuous
        if len(np.unique(y)) == 1:
            self.population = []
            self.weights = []
            for val in range(np.max(y) + 1):
                self.population.append(val)
                sum = 0
                for v in y:
                    if v == val:
                        sum += 1
                self.weights.append(sum/len(y))
            return
        greatest_reduction = -np.inf
        best_split_feature = 0
        best_split_value = 0
        reduction, split_value = 0, 0
        for feature_idx in range(len(x[0])):
            feature_arr = None
            if features[feature_idx] == "cont":
                split_point = DecisionTree.find_split_point(x[:, feature_idx], y)
                feature_arr = np.zeros(len(x[:, feature_idx]))
                for idx in range(len(feature_arr)):
                    if x[:, feature_idx][idx] > split_point:
                        feature_arr[idx] = 1
                reduction, split_value = DecisionTree.gini_reduction(feature_arr, y, True, split_point)
            else:
                feature_arr = x[:, feature_idx]
                reduction, split_value = DecisionTree.gini_reduction(feature_arr, y)
            if reduction > greatest_reduction:
                greatest_reduction = reduction
                best_split_feature = feature_idx
                best_split_value = split_value
        self.split_feature = best_split_feature
        self.split_value = best_split_value
        self.split_feature_type = self.features[self.split_feature]

        self.left = DecisionTree()
        self.right = DecisionTree()
        df = pd.DataFrame(x)
        df[len(df.columns)] = y

        match_df = None
        not_match_df = None
        if self.split_feature_type == "cont":
            match_df = df[df[self.split_feature] < self.split_value]
            not_match_df = df[df[self.split_feature] >= self.split_value]
        else:
            match_df = df[df[self.split_feature] == self.split_value]
            not_match_df = df[df[self.split_feature] != self.split_value]
        print(len(match_df), len(not_match_df))
        self.left.init(match_df.loc[:, :len(df.columns)-2].values, match_df.loc[:, len(df.columns)-1].values, self.features)
        self.right.init(not_match_df.loc[:, :len(df.columns)-2].values, not_match_df.loc[:, len(df.columns)-1].values, self.features)

    def find_split_point(feature_col, y):
        max_info_gain = -np.inf
        best_split_point = 1
        feature_df = pd.DataFrame(feature_col)
        feature_df[1] = y
        feature_df = feature_df.sort_values(0)

        entropy = DecisionTree.info_entropy(feature_df[1])

        for split_idx in range(0, len(feature_col) - 1):
            split_point = (feature_df[0][split_idx] + feature_df[0][split_idx + 1]) / 2
            l = feature_df.loc[:split_idx,:]
            r = feature_df.loc[split_idx + 1:,:]
            info_gain_val = entropy - DecisionTree.info_needed(l, r)
            if info_gain_val > max_info_gain:
                max_info_gain = info_gain_val
                best_split_point = split_point
        return best_split_point

    def info_entropy(y_col):
        frequency_col = y_col.value_counts()
        frequency_col = frequency_col[frequency_col != 0]
        total = np.sum(frequency_col.values)
        frequency_col = frequency_col/total * np.log2(frequency_col/total)
        return -np.sum(frequency_col)

    def info_needed(l_df, r_df):
        total = len(l_df) + len(r_df)
        info_needed = 0
        info_needed += len(l_df)/total * DecisionTree.info_entropy(l_df[1])
        info_needed += len(r_df)/total * DecisionTree.info_entropy(r_df[1])
        return info_needed

    def gini_index(df):
        total = len(df)
        ret = 1
        for y in range(np.max(df[1]) + 1):
            ret -= (len(df[df[1] == y])/total)**2
        return ret

    def gini_reduction(feature_arr, y, is_cont=False, split_val=None):
        df = pd.DataFrame(feature_arr)
        total_vals = len(df)
        df[1] = y
        total_gini_index = DecisionTree.gini_index(df)

        greatest_reduction = -np.inf
        best_split_value = 0
        for possible_value in range(int(np.max(feature_arr)) + 1):
            match_df = df[df[0] == possible_value]
            not_match_df = df[df[0] != possible_value]

            gini_index = len(match_df)/total_vals * DecisionTree.gini_index(match_df) + len(not_match_df)/total_vals * DecisionTree.gini_index(not_match_df)
            gini_reduction = total_gini_index - gini_index
            if gini_reduction > greatest_reduction:
                greatest_reduction = gini_reduction
                best_split_value = possible_value
        if is_cont:
            return greatest_reduction, split_val
        return greatest_reduction, best_split_value

    def predict(self, x):
        if self.left == None and self.right == None:
            return random.choices(population=self.population, weights=self.weights)[0]
        if self.split_feature_type == "cont":
            if x[self.split_feature] < self.split_value:
                return self.left.predict(x)
            else:
                return self.right.predict(x)
        else:
            if x[self.split_feature] == self.split_value:
                return self.left.predict(x)
            else:
                return self.right.predict(x)
