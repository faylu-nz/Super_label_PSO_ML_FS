import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from skmultilearn.adapt import MLkNN
import warnings
warnings.filterwarnings('ignore')


"""
In each subcgroup(cluster), get label indexes
"""
def ClusterIndicesNumpy(clustNum, labels_array):  # numpy
    return np.where(labels_array == clustNum)[0]



"""
Convert each subgroup's labels(binary nparray) into super label(list)
If all labels are 0, super label is 0; otherwise, super label is assigned to 1
"""
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




"""
Convert original y labels into y_super labels
subgroup_labels are dataframe with original labels + super labels
"""
def label_convert_simple(y, no_cls):  # Here y is ndarray, need to convert to df

    super_labels = []
    kmeans = KMeans(n_clusters=no_cls, random_state=0).fit(y.T)

    y_df = pd.DataFrame(y)
    for i in range(no_cls):
        # Column numbers(indexes) of all the labels in each cluster
        cluster = ClusterIndicesNumpy(i, kmeans.labels_)
        # Get all the original labels from cluster, dataframe form
        subgroup_label = y_df.iloc[:, cluster]
        # Convert original labels to a column super label
        s = pd.DataFrame(convert(subgroup_label), columns=['s' + str(i)])
        super_labels.append(s)

    # Combine all super label columns, as orginal y converted to super_label y, the target
    y_s = pd.concat(super_labels, axis=1)
    y_s = y_s.to_numpy()

    return y_s



"""
Get trained super_classifier
"""
def super_classifier(X_train, y_train_s):

    clf_super = MLkNN(k=3)
    scaler_1 = StandardScaler()
    scaler_2 = MinMaxScaler()
    X_train_scaled_1 = scaler_1.fit_transform(X_train)
    X_train_scaled_2 = scaler_2.fit_transform(X_train_scaled_1)
    clf_super.fit(X_train_scaled_2, y_train_s)

    return clf_super



"""
Do super classification on dataset
"""
def super_classification(clf_super, X_test):

    scaler_1 = StandardScaler()
    scaler_2 = MinMaxScaler()
    X_test_scaled_1 = scaler_1.fit_transform(X_test)
    X_test_scaled_2 = scaler_2.fit_transform(X_test_scaled_1)
    y_test_s_pred = clf_super.predict(X_test_scaled_2).toarray()
    return y_test_s_pred
    
    
