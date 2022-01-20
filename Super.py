import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from skmultilearn.adapt import MLkNN
import warnings
warnings.filterwarnings('ignore')


# In each subcgroup(cluster), get label indexes

def ClusterIndicesNumpy(clustNum, labels_array):  # numpy
    return np.where(labels_array == clustNum)[0]


# Convert each subgroup's labels(binary nparray) into super label(list)
# If all labels are 0, super label is 0; otherwise, super label is assigned to 1

def convert(subgroup_label):
    super_ = []
    subgroup_label_array = subgroup_label.to_numpy()
    rows = subgroup_label.shape[0]
    columns = subgroup_label.shape[1]
    for row in range(rows):
        s = 0
        for column in range(columns):
            if subgroup_label_array[row][column] == 1:
                s = 1
                break
        super_.append(s)
    return super_


# Convert original y labels into y_super labels
# subgroup_labels are dataframe with original labels + super labels

def label_convert(y_train_, no_cls):  # Here y is dataframe

    subgroups = []

    super_labels = []
    kmeans = KMeans(n_clusters=no_cls, random_state=0).fit(y_train_.T)
    # dictionary to record key(cluster index) and value(cluster columns)
    dict_clst_col = dict()

    for i in range(no_cls):
        # Column numbers(indexes) of all the labels in each cluster
        cluster = ClusterIndicesNumpy(i, kmeans.labels_)
        dict_clst_col[i] = cluster
        # Get all the original labels from cluster, dataframe form
        subgroup_label = y_train_.iloc[:, cluster]
        # Convert original labels to a column super label
        s = pd.DataFrame(convert(subgroup_label), columns=['s' + str(i)])
        super_labels.append(s)
        subgroup_label['s' + str(i)] = s   # Concat s into subgroup
        subgroups.append(subgroup_label)

    # Combine all super label columns, as orginal y converted to super_label y, the target
    y_s = pd.concat(super_labels, axis=1)
    y_s = y_s.to_numpy()

    return y_s, subgroups, dict_clst_col


# After split, indexes of X_train, X_test and y_train, y_test will be discorder, aka not ascending any more.
# If index disorder, will be tricky to process index, e.g. zero_idx
# Need to reorder index first.

def convert_index(X_train, X_test, y_train, y_test):

    X_train_ = X_train.reset_index(drop=True)
    y_train_ = y_train.reset_index(drop=True)
    X_test_ = X_test.reset_index(drop=True)
    y_test_ = y_test.reset_index(drop=True)

    return X_train_, X_test_, y_train_, y_test_


# Get trained super_classifier

# Get trained super_classifier

def super_classifier(X_train_, y_train_, no_cls):

    clf = MLkNN(k=3)

    scaler_1 = StandardScaler()
    scaler_2 = MinMaxScaler()
    X_train_scaled_1 = scaler_1.fit_transform(X_train_)
    X_train_scaled_2 = scaler_2.fit_transform(X_train_scaled_1)
#     scaler = StandardScaler()
#     X_train_ = scaler.fit_transform(X_train_)
#     X_train_ = MinMaxScaler().fit_transform(X_train_)
    y_s, subgroups, dict_clst_col = label_convert(y_train_, no_cls)
    clf.fit(X_train_scaled_2, y_s)
#     clf.fit(X_train_, y_s)

    return clf


# For each subgroup, collect all the zero super labels
# The len of total_zeros is the number of subgroups, also the number of super labels

def zeros(y_s):

    total_zeros = []
    for i in range(y_s.shape[1]):     # number of super labels
        idx_zeros = []
        for j in range(y_s.shape[0]):   # number of instances
            if y_s[j][i] == 0:
                idx_zeros.append(j)
        total_zeros.append(idx_zeros)

    return total_zeros


# In each subgroup, if a particular row of y_s is zero, the corresponding X features also needs to be removed.
# For each subgroup, get the indexes of zeros in one y_s, and remove these same indexes from X feature instances.
# Each subgroup contains its own X, means different subgroup contain diffenrent number of instances
# Collect each removed X and return.

def remove_zeros(X, y_s):  # y_s is ndarray
    total_zeros = zeros(y_s)
    Xs = []
    for idx_zeros in total_zeros:
        X_ = pd.DataFrame(X).drop(idx_zeros)
        Xs.append(X_)
    return Xs

    """
From original X and y, compute super label y(y_s), which actually comes from true y. 
Each subgroup contains original y labels + y_s label. 
Check each y_s, if 0, than remove the whole line, which means remove its corresponding original labels, and its X. 
So the remaining of original labels, as well as X of each subgroup are different, since indexes of zeros in each y super label are different.

def sub_classification is for each subgroup, train X_(X remove y_s's zero indexes) and y_(y sub original labels remove y_s's zero indexes). After training, collect all sub-clfs and Xs.
    """


# Get trained sub classifiers

def sub_classifiers(X_train_, y_train_, no_cls):

    clfs = []
    n_sub_labelses = []
    # y_super labels, converted from original y labels (target)
    y_s, subgroups, dict_clst_col = label_convert(y_train_, no_cls)
    total_zeros = zeros(y_s)
#     scaler = StandardScaler()
#     scaler = MinMaxScaler()
    scaler_1 = StandardScaler()
    scaler_2 = MinMaxScaler()
    Xs = remove_zeros(X_train_, y_s)

    for subgroup, idx_zeros, X in zip(subgroups, total_zeros, Xs):
        # Have different X, because idx of zero are different
        # Drop all the zero instances, both in X and y, aka X_, y_
        y_ = subgroup.drop(idx_zeros)
        y_ = y_.drop(y_.columns[-1:], axis=1)  # Remove the s label
        n_sub_labels = subgroup.shape[1]-1
        n_sub_labelses.append(n_sub_labels)

        X_scaled_1 = scaler_1.fit_transform(X)
        X_scaled_2 = scaler_2.fit_transform(X_scaled_1)
#         X_ = scaler.fit_transform(X)
        clf = MLkNN(k=3)
#         clf.fit(X_, y_.to_numpy())
        clf.fit(X_scaled_2, y_.to_numpy())

        clfs.append(clf)

    return clfs, n_sub_labelses

    """
Now already gained clf, which is classifier for super classification, and clfs which are for all the sub-classifications. 
Then will apply clf and clfs on training set, to see the training_loss, and then apply on test set, to get test_loss. 
Finally, compare.
Apply clf, clfs, Xs on training set.
    """

# Do super classification on dataset


def super_classification(clf_super, X_test_):
    #     scaler = StandardScaler()
    scaler_1 = StandardScaler()
    scaler_2 = MinMaxScaler()
    X_test_scaled_1 = scaler_1.fit_transform(X_test_)
    X_test_scaled_2 = scaler_2.fit_transform(X_test_scaled_1)
#     X_test_scaled = scaler.fit_transform(X_test_)
#     X_test_scaled = MinMaxScaler().fit_transform(X_test_)
#     y_test_s_pred = clf_super.predict(X_test_scaled).toarray()   # Predicted super labels, will be passed into def zeros().
    y_test_s_pred = clf_super.predict(X_test_scaled_2).toarray()
    return y_test_s_pred


# Do sub-classification on sub-datasets (original X + subgroup original labels)

def sub_classification(clfs, n_sub_labelses, X_test_, y_test_s_pred):

    # Based on predicted super label, compute which are zeros in each subgroup
    total_test_zeros = zeros(y_test_s_pred)

    # Remove zeros in each subgroup in X
    X_tests = remove_zeros(X_test_, y_test_s_pred)

    y_test_sub_preds = []
#     scaler = StandardScaler()
#     scaler = MinMaxScaler()
    scaler_1 = StandardScaler()
    scaler_2 = MinMaxScaler()
    for clf, X_test in zip(clfs, X_tests):
        # based on corrrespanding y_s_pred column, the y_labels in column are all zeros
        if len(X_test.axes[0]) == 0:
            # so when remove zeros in X_test, n_rowsof X-test is 0, aka no intances to predict
            y_test_sub_pred = None
            y_test_sub_preds.append(y_test_sub_pred)
        else:
            #             X_scaled = scaler.fit_transform(X_test)
            X_scaled_1 = scaler_1.fit_transform(X_test)
            X_scaled_2 = scaler_2.fit_transform(X_scaled_1)
#             y_test_sub_pred = clf.predict(X_scaled)   # y_test_sub_pred is sparse matrix
            y_test_sub_pred = clf.predict(X_scaled_2)
            y_test_sub_preds.append(y_test_sub_pred)

    # total_test_zeros, y_test_sub_labels are lists
    return total_test_zeros, y_test_sub_preds

    """
After super and sub-classifications are done on test/validation dataset, the next step is to revert and rebuild all the predicted sub-labels together. 
The predicted subgroups do not contain all the original instances, coz those all-zeros instances are removed before sub-classification. 
So when doing revert, we need to find out which instances are all-zeros(those predicted super-predicted are zero), these can be reverted to [0,0,0,...]. 
If the super-predicted label is not 0, then this predicted instance's subgroup labels are in coreesponding y_sub_pred.

len(total_zeros) is number of subgroups, aka number of columns for super_y_labels for each subgroup, y.shape[0] is the rows, aka instances in original y, 
if index of the instance is included in column in total_zeros, that means when revert to original labels, 
we can fill out all the subgroup labels of this instance to all zeros

How to fill out the zero super label's corresponding sub-labels to zeros? 
We need to know how many sub-labels in each cluster. 
That is how many labels in each y_sub_pred
    """

# Fill out zeros back into each y_sub_pred


def fill_zeros(total_test_zeros, y_test_sub_preds, y_test_, n_sub_labelses):
    total_filled_preds = []

    for i in range(len(total_test_zeros)):   # number of columns of y_super
        labels = []
        if y_test_sub_preds[i] == None:
            for j in range(y_test_.shape[0]):
                # Create all zero list, then list of list, append
                labels.append(np.zeros(n_sub_labelses[i], dtype=np.int64))
        else:
            # y_test_sub_pred is sparse matrix, convert to nparray
            y_test_sub_pred_np = y_test_sub_preds[i].toarray()
            no_sublabels = y_test_sub_pred_np.shape[1]
            for j in range(y_test_.shape[0]):   # original y_test's rows
                if j in total_test_zeros[i]:
                    labels.append(np.zeros(no_sublabels, dtype=np.int64))
                else:
                    label = y_test_sub_pred_np[0]
                    labels.append(label)
                    y_test_sub_pred_np = np.delete(
                        y_test_sub_pred_np, [0], axis=0)
        total_filled_preds.append(labels)

    return total_filled_preds


# sort the order of y_test_preds, aka revert to original order as y_test's columns(before clustering)
# Clusters are not neccessarily equal, so could convert total_filled_preds to np.array

def sort_preds(total_filled_preds, y_train_, no_cls):

    y_s, subgroup, dict_clst_col = label_convert(y_train_, no_cls)
    y_test_sub_preds = []
    for i in range(len(total_filled_preds)):
        np_total_filled_pred = np.array(total_filled_preds[i])
        y_test_sub_preds.append(pd.DataFrame(
            np_total_filled_pred, columns=dict_clst_col[i]))
    y_test_pred = pd.concat(y_test_sub_preds, axis=1)
    y_test_pred_t = y_test_pred.T
    y_test_pred_t_sorted = y_test_pred_t.sort_index(ascending=True)
    y_test_pred_sorted = y_test_pred_t_sorted.T

    return y_test_pred_sorted


# Based on splited X_train, y_train, calculate the y_test_pred on x_test

def calc_preds(X_train, y_train, X_test, y_test, no_cls):

    X_train_, X_test_, y_train_, y_test_ = convert_index(
        X_train, X_test, y_train, y_test)

    clf_super = super_classifier(X_train_, y_train_, no_cls)
    clfs_sub, n_sub_labelses = sub_classifiers(X_train_, y_train_, no_cls)
    y_test_s_pred = super_classification(clf_super, X_test_)
    total_test_zeros, y_test_sub_preds = sub_classification(
        clfs_sub, n_sub_labelses, X_test_, y_test_s_pred)
    total_filled_preds = fill_zeros(
        total_test_zeros, y_test_sub_preds, y_test_, n_sub_labelses)
    y_test_pred_sorted = sort_preds(total_filled_preds, y_train_, no_cls)

    return y_test_pred_sorted, y_test_
