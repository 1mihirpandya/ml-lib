import numpy as np
import pandas as pd
import random
import math
import time

class DecisionTree:
    def __init__(self):
        self.left, self.right = None, None
        self.features = []

    def init(self, df, features):
        self.features = features #cat=categorical or cont=continuous
        y_idx = df.columns[-1]
        if len(np.unique(df[y_idx])) == 1:
            self.population = []
            self.weights = []
            for val in range(np.max(df[y_idx]) + 1):
                self.population.append(val)
                sum = 0
                for v in df[y_idx]:
                    if v == val:
                        sum += 1
                self.weights.append(sum/len(df[y_idx]))
            return
        greatest_reduction = -np.inf
        best_split_feature = 0
        best_split_value = 0
        reduction, split_value = 0, 0
        for feature_idx in range(len(df.columns) - 1):
            if features[feature_idx] == "cont":
                reduction, split_value = DecisionTree.find_split_point(df[[feature_idx, y_idx]])
            else:
                reduction, split_value = DecisionTree.gini_reduction_cat(df[[feature_idx, y_idx]])
            if reduction > greatest_reduction:
                greatest_reduction = reduction
                best_split_feature = feature_idx
                best_split_value = split_value

        self.split_feature = best_split_feature
        self.split_value = best_split_value
        self.split_feature_type = self.features[self.split_feature]

        self.left = DecisionTree()
        self.right = DecisionTree()

        match_df = None
        not_match_df = None
        if self.split_feature_type == "cont":
            match_df = df[df[self.split_feature] < self.split_value]
            not_match_df = df[df[self.split_feature] > self.split_value]
        else:
            match_df = df[df[self.split_feature] == self.split_value]
            not_match_df = df[df[self.split_feature] != self.split_value]

        self.left.init(match_df, self.features)
        self.right.init(not_match_df, self.features)

    def find_split_point(feature_df):
        col_idx = feature_df.columns[0]
        y_idx = feature_df.columns[-1]
        best_reduction = -np.inf
        best_split_point = 1
        feature_df = feature_df.sort_values(col_idx)
        feature_df = feature_df.reset_index(drop=True)

        gini = DecisionTree.gini_index(feature_df)
        left_count, right_count = 0, len(feature_df)
        card = np.max(feature_df[y_idx]) + 1
        left, right = np.zeros(card), np.zeros(card)

        for idx in range(0, len(feature_df)):
            right[int(feature_df[y_idx][idx])] += 1

        for split_idx in range(0, len(feature_df) - 1):
            split_point = (feature_df[col_idx][split_idx] + feature_df[col_idx][split_idx + 1]) / 2
            if split_point == feature_df[col_idx][split_idx]:
                continue
            left_count += 1
            right_count -= 1
            left[int(feature_df[y_idx][split_idx])] += 1
            right[int(feature_df[y_idx][split_idx])] -= 1

            gini_reduction = left_count / (left_count + right_count) * (1 - np.sum([(freq/left_count)**2 for freq in left]))
            gini_reduction += right_count / (left_count + right_count) * (1 - np.sum([(freq/right_count)**2 for freq in right]))
            gini_reduction = gini - gini_reduction
            if gini_reduction > best_reduction:
                best_reduction = gini_reduction
                best_split_point = split_point
        return best_reduction, best_split_point

    def info_entropy(y_col):
        frequency_col = y_col.value_counts()
        frequency_col = frequency_col[frequency_col != 0]
        total = np.sum(frequency_col.values)
        frequency_col = frequency_col/total * np.log2(frequency_col/total)
        return -np.sum(frequency_col)

    def info_needed(l_df, r_df):
        total = len(l_df) + len(r_df)
        info_needed = 0
        info_needed += len(l_df)/total * DecisionTree.info_entropy(l_df[l_df.columns[-1]])
        info_needed += len(r_df)/total * DecisionTree.info_entropy(r_df[r_df.columns[-1]])
        return info_needed

    def gini_index(df):
        total = len(df)
        ret = 1
        for y in range(np.max(df[df.columns[-1]]) + 1):
            ret -= (len(df[df[df.columns[-1]] == y])/total)**2
        return ret

    # def gini_reduction_cont(df, split_val):
    #     total_vals = len(df)
    #     match_df = df[df[df.columns[0]] < split_val]
    #     not_match_df = df[df[df.columns[0]] > split_val]
    #     gini_index = len(match_df)/total_vals * DecisionTree.gini_index(match_df) + len(not_match_df)/total_vals * DecisionTree.gini_index(not_match_df)
    #     gini_reduction = gini_index
    #     return gini_reduction, split_val

    def gini_reduction_cat(df):
        total_vals = len(df)
        total_gini_index = DecisionTree.gini_index(df)

        greatest_reduction = -np.inf
        best_split_value = 0
        for possible_value in range(np.max(df[df.columns[0]]) + 1):
            match_df = df[df[df.columns[0]] == possible_value]
            not_match_df = df[df[df.columns[0]] != possible_value]
            if len(match_df) == 0 or len(not_match_df) == 0:
                continue

            gini_index = len(match_df)/total_vals * DecisionTree.gini_index(match_df) + len(not_match_df)/total_vals * DecisionTree.gini_index(not_match_df)
            gini_reduction = total_gini_index - gini_index
            if gini_reduction > greatest_reduction:
                greatest_reduction = gini_reduction
                best_split_value = possible_value
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
