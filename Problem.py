import Super
import Util
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skmultilearn.adapt import MLkNN
from skmultilearn.model_selection import IterativeStratification
import warnings
warnings.filterwarnings('ignore')


class Problem:
    def __init__(self, minimize):
        self.minimize = minimize

    def fitness(self, solution):
        return 1

    def is_better(self, first, second):
        if self.minimize == True:
            return first < second
        else:
            return first > second

    def worst_fitness(self):
        if self.minimize == True:
            return float('inf')
        else:
            return float('-inf')


class FS(Problem):

    def __init__(self, minimize, X, y):
        self.minimize = minimize,
        self.X = X
        self.y = y
        self.threshold = 0.6

    def fitness(self, solution):
        feature_selected = np.where(solution > self.threshold)[0]
        X = self.X[:, feature_selected]
        y = self.y
        if len(feature_selected) == 0:
            return self.worst_fitness()
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

        clf = KNN()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_pred, y_test)
        return acc


# Standard PSO
# Fitness is MLKNN classification hamming loss.

class FS_ML(Problem):

    def __init__(self, minimize, X, y):
        self.minimize = minimize,
        self.X = X
        self.y = y
        self.threshold = 0.6

    def fitness(self, solution):
        feature_selected = np.where(solution > self.threshold)[0]
        X = self.X[:, feature_selected]
        y = self.y
        if len(feature_selected) == 0:
            return self.worst_fitness()

        n_splits = 5
        clf = MLkNN(k=3)
        scaler = StandardScaler()
        k_fold = IterativeStratification(n_splits=n_splits, order=1)

        hamming_losses = 0
        for train_idx, test_idx in k_fold.split(X, y):

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            ham = hamming_loss(y_test, y_pred)
            hamming_losses += ham

        return hamming_losses/n_splits


# Super label PSO
# Fitness is use super+sub classification hamming loss
# 0109

class FS_ML_super_hl(Problem):

    def __init__(self, minimize, X, y):
        self.minimize = minimize,
        self.X = X
        self.y = y
        self.threshold = 0.6

    def fitness(self, solution):
        feature_selected = np.where(solution > self.threshold)[0]
        X = self.X[:, feature_selected]
        y = self.y
        if len(feature_selected) == 0:
            return self.worst_fitness()

        n_splits = 5
        k_fold = IterativeStratification(n_splits=n_splits, order=1)

        hamming_losses = 0
        for train_idx, test_idx in k_fold.split(X, y):

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            X_train_df, X_test_df, y_train_df, y_test_df = Util.Convert_to_df(
                X_train, X_test, y_train, y_test)
            # y_s, subgroups, dict_clst_col = Super.label_convert(y_train_df, 2)
            # print(dict_clst_col)
            y_test_pred, y_test = Super.calc_preds(
                X_train_df, y_train_df, X_test_df, y_test_df)

            ham = hamming_loss(y_test, y_test_pred)
            hamming_losses += ham

        return hamming_losses/n_splits

        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, test_size=0.33, random_state=42)
        # X_train_df, X_test_df, y_train_df, y_test_df = Util.Convert_to_df(
        #     X_train, X_test, y_train, y_test)
        # y_test_pred, y_test = Super.calc_preds(
        #     X_train_df, y_train_df, X_test_df, y_test_df)
        # ham = hamming_loss(y_test, y_test_pred)
        # return ham

        # Super label PSO
# Fitness is use super+sub classification f1 score micro
# modified 0120 add no_cls


class FS_ML_super(Problem):

    def __init__(self, minimize, X, y, no_cls):
        self.minimize = minimize,
        self.X = X
        self.y = y
        self.no_cls = no_cls   # modified 0120 add no_cls
        self.threshold = 0.6

    def fitness(self, solution):
        feature_selected = np.where(solution > self.threshold)[0]
        X = self.X[:, feature_selected]
        y = self.y
        no_cls = self.no_cls  # modified 0120 add no_cls
        if len(feature_selected) == 0:
            return self.worst_fitness()

        n_splits = 5
        k_fold = IterativeStratification(
            n_splits=n_splits, order=1, random_state=42)

        f1_mics = 0
        for train_idx, test_idx in k_fold.split(X, y):

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            X_train_df, X_test_df, y_train_df, y_test_df = Util.Convert_to_df(
                X_train, X_test, y_train, y_test)
            y_test_pred, y_test = Super.calc_preds(
                X_train_df, y_train_df, X_test_df, y_test_df, no_cls)

            f1_mic = f1_score(y_test, y_test_pred, average='micro')
            f1_mics += f1_mic

        return f1_mics/n_splits
