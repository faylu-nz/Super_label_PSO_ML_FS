import Problem
import Swarm
import Util
import Super
import numpy as np
import pandas as pd
from sklearn.metrics import hamming_loss
from sklearn.preprocessing import StandardScaler
from skmultilearn.adapt import MLkNN
from skmultilearn.model_selection import IterativeStratification
from skmultilearn.dataset import load_dataset

import time
import warnings
warnings.filterwarnings('ignore')


def full_std_full_sup():
    # data = Util.read_arff('datasets/' + datasets[2] + '.arff')

    # y = data.iloc[:, n_featureses[2]:]
    # X = data.iloc[:, :n_featureses[2]]

    # n_features = len(list(X))
    # X = X.to_numpy()
    # y = y.to_numpy()

    X, y, feature_names, label_names = load_dataset(
        datasets_large[4], 'undivided')
    X = pd.DataFrame.sparse.from_spmatrix(X).to_numpy()
    y = pd.DataFrame.sparse.from_spmatrix(y).to_numpy()

    n_splits = 5
    k_fold = IterativeStratification(
        n_splits=n_splits, order=1, random_state=42)

    standard_hams = []
    super_hams = []
    f_ratios = []
    to_print = ''
    fold_count = 0

    for train_idx, test_idx in k_fold.split(X, y):
        fold_count += 1

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # get super classification acc(hl)
        dict_cls_ham = dict()   # no_cls:super_ham dictionary
        for no_cls in no_clses_large:
            X_train_df, X_test_df, y_train_df, y_test_df = Util.Convert_to_df(
                X_train, X_test, y_train, y_test)
            y_test_pred_super, y_test_super = Super.calc_preds(
                X_train_df, y_train_df, X_test_df, y_test_df, no_cls)
            super_ham = hamming_loss(y_test_super, y_test_pred_super)
            dict_cls_ham[no_cls] = super_ham
            super_hams.append(dict_cls_ham)

        # get standard classficaition acc(hl)
        scaler = StandardScaler()
    #     scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.fit_transform(X_test)

        clf = MLkNN(k=3)
        clf.fit(X_train_scaled, y_train)
        y_test_pred = clf.predict(X_test_scaled)
        clf.fit(X_train, y_train)
        y_test_pred = clf.predict(X_test)
        standard_ham = hamming_loss(y_test, y_test_pred)
        standard_hams.append(standard_ham)

        # to write the results
        to_print += '--------------Fold %d----------------\n' % fold_count
        to_print += 'Standard classification with full feature hamming loss: %.4f\n' % standard_ham
        to_print += 'Super classification with full feature hamming loss with 2 clusters: %.4f\n' % dict_cls_ham[2]
        to_print += 'Super classification with full feature hamming loss with 4 clusters: %.4f\n' % dict_cls_ham[4]
        to_print += 'Super classification with full feature hamming loss with 6 clusters: %.4f\n' % dict_cls_ham[6]
        to_print += 'Super classification with full feature hamming loss with 8 clusters: %.4f\n' % dict_cls_ham[8]
        to_print += 'Super classification with full feature hamming loss with 10 clusters: %.4f\n' % dict_cls_ham[10]

    to_print += '--------------Average----------------\n'
    to_print += 'Ave Standard Classification Accuracy: %.4f\n' % np.average(
        standard_hams)
    # to_print += 'Ave Super Classification Accuracy: %.4f\n' % np.average(super_hams)
    to_print += 'Ave Super Classification with 2 clusters: %.4f\n' % np.average(
        [super_hams[0][2], super_hams[1][2], super_hams[2][2], super_hams[3][2], super_hams[4][2]])
    to_print += 'Ave Super Classification with 4 clusters: %.4f\n' % np.average(
        [super_hams[0][4], super_hams[1][4], super_hams[2][4], super_hams[3][4], super_hams[4][4]])
    to_print += 'Ave Super Classification with 6 clusters: %.4f\n' % np.average(
        [super_hams[0][6], super_hams[1][6], super_hams[2][6], super_hams[3][6], super_hams[4][6]])
    to_print += 'Ave Super Classification with 8 clusters: %.4f\n' % np.average(
        [super_hams[0][8], super_hams[1][8], super_hams[2][8], super_hams[3][8], super_hams[4][8]])
    to_print += 'Ave Super Classification with 10 clusters: %.4f\n' % np.average(
        [super_hams[0][10], super_hams[1][10], super_hams[2][10], super_hams[3][10], super_hams[4][10]])

    f = open('records/record_stdscaler_' +
             datasets_large[4] + '_full_standard_super_clf.txt', 'w')
    f.write(to_print)
    f.close()


def full_std_sel_sup():
    data = Util.read_arff('datasets/' + datasets[6] + '.arff')

    # y = data.iloc[:, locations[6]:]
    # X = data.iloc[:, :locations[6]]
    # n_features = len(list(X))
    # X = X.to_numpy()
    # y = y.to_numpy()

    X, y, feature_names, label_names = load_dataset(
        datasets_large[0], 'undivided')
    X = pd.DataFrame.sparse.from_spmatrix(X).to_numpy()
    y = pd.DataFrame.sparse.from_spmatrix(y).to_numpy()
    n_features = len(list(X))

    n_splits = 5
    k_fold = IterativeStratification(
        n_splits=n_splits, order=1, random_state=42)

    full_hams = []
    sel_hams = []
    PSO_durations = []
    f_ratios = []
    to_print = ''
    fold_count = 0

    for train_idx, test_idx in k_fold.split(X, y):
        print('Fold ', fold_count)
        fold_count += 1

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # get full acc
        scaler = StandardScaler()
    #     scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.fit_transform(X_test)

        clf = MLkNN(k=3)
        clf.fit(X_train_scaled, y_train)
        y_test_pred = clf.predict(X_test_scaled)
        full_ham = hamming_loss(y_test, y_test_pred)
        full_hams.append(full_ham)

        #  perform FS
        start_PSO = time.time_ns()    # marking start time of PSO
    #        problem = FS_ML(minimize=True, X=X_train, y=y_train)
        problem = Problem.FS_ML_super(minimize=True, X=X_train, y=y_train)

        # parameter for PSO
        pop_size = 3
        n_iterations = 5
        swarm = Swarm(n_particles=pop_size, length=n_features, pos_max=1.0, pos_min=0,
                      vel_max=0.2, vel_min=-0.2, problem=problem, n_iterations=n_iterations)
        best_sol, best_fit = swarm.iterate()
        end_PSO = time.time_ns()      # marking ending time of PSO
        duration_PSO = round((end_PSO - start_PSO)/1000000000, 2)
        PSO_durations.append(duration_PSO)

        # process the final solution
        sel_fea = np.where(best_sol > problem.threshold)[0]
        clf.fit(X_train[:, sel_fea], y_train)
        y_test_pred = clf.predict(X_test[:, sel_fea])
        fold_ham = hamming_loss(y_true=y_test, y_pred=y_test_pred)
        sel_hams.append(fold_ham)
        f_ratios.append(len(sel_fea)/n_features)

        # to write the results
        to_print += '--------------Fold %d----------------\n' % fold_count
        to_print += 'Full feature hamming loss: %.4f\n' % full_ham
        to_print += 'Fold selected hamming loss: %.4f\n' % fold_ham
        to_print += 'Time of PSO: %.4f\n' % duration_PSO
        to_print += 'Selection ratio: %.2f\n' % (len(sel_fea)/n_features)
        to_print += 'Selected features: %s\n' % (
            ', '.join([str(ele) for ele in sel_fea]))

    to_print += '--------------Average----------------\n'
    to_print += 'Ave Full Accuracy: %.4f\n' % np.average(full_hams)
    to_print += 'Ave Selection Accuracy: %.4f\n' % np.average(sel_hams)
    to_print += 'Ave time of PSO: %.4f\n' % np.average(PSO_durations)
    to_print += 'Ave Feature Ratio: %.2f\n' % np.average(f_ratios)

    f = open('records/record_' + datasets[6] + '_super_PSO.txt', 'w')
    f.write(to_print)
    f.close()


def sel_std_sel_sup():
    data = read_arff('datasets/' + datasets[6] + '.arff')

    y = data.iloc[:, locations[6]:]
    X = data.iloc[:, :locations[6]]

    n_features = len(list(X))
    X = X.to_numpy()
    y = y.to_numpy()

    n_splits = 5
    k_fold = IterativeStratification(
        n_splits=n_splits, order=1, random_state=42)

    full_hams = []
    sel_hams = []
    PSO_durations = []
    f_ratios = []
    to_print = ''
    fold_count = 0

    for train_idx, test_idx in k_fold.split(X, y):
        print('Fold ', fold_count)
        fold_count += 1

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # get full acc
        scaler = StandardScaler()
    #     scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.fit_transform(X_test)

        clf = MLkNN(k=3)
        clf.fit(X_train_scaled, y_train)
        y_test_pred = clf.predict(X_test_scaled)
        full_ham = hamming_loss(y_test, y_test_pred)
        full_hams.append(full_ham)

        #  perform FS
        start_PSO = time.time_ns()    # marking start time of PSO
        problem = FS_ML(minimize=True, X=X_train, y=y_train)
    #     problem = FS_ML_super(minimize=True, X=X_train, y=y_train)

        # parameter for PSO
        pop_size = 3
        n_iterations = 5
        swarm = Swarm(n_particles=pop_size, length=n_features, pos_max=1.0, pos_min=0,
                      vel_max=0.2, vel_min=-0.2, problem=problem, n_iterations=n_iterations)
        best_sol, best_fit = swarm.iterate()
        end_PSO = time.time_ns()      # marking ending time of PSO
        duration_PSO = round((end_PSO - start_PSO)/1000000000, 2)
        PSO_durations.append(duration_PSO)

        # process the final solution
        sel_fea = np.where(best_sol > problem.threshold)[0]
        clf.fit(X_train[:, sel_fea], y_train)
        y_test_pred = clf.predict(X_test[:, sel_fea])
        fold_ham = hamming_loss(y_true=y_test, y_pred=y_test_pred)
        sel_hams.append(fold_ham)
        f_ratios.append(len(sel_fea)/n_features)

        # to write the results
        to_print += '--------------Fold %d----------------\n' % fold_count
        to_print += 'Full feature hamming loss: %.4f\n' % full_ham
        to_print += 'Fold selected hamming loss: %.4f\n' % fold_ham
        to_print += 'Time of PSO: %.4f\n' % duration_PSO
        to_print += 'Selection ratio: %.2f\n' % (len(sel_fea)/n_features)
        to_print += 'Selected features: %s\n' % (
            ', '.join([str(ele) for ele in sel_fea]))

    to_print += '--------------Average----------------\n'
    to_print += 'Ave Full Accuracy: %.4f\n' % np.average(full_hams)
    to_print += 'Ave Selection Accuracy: %.4f\n' % np.average(sel_hams)
    to_print += 'Ave time of PSO: %.4f\n' % np.average(PSO_durations)
    to_print += 'Ave Feature Ratio: %.2f\n' % np.average(f_ratios)

    f = open('records/record_' + datasets[6] + '_standardPSO.txt', 'w')
    f.write(to_print)
    f.close()
