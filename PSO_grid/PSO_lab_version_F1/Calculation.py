
import Problem
import Swarm
import Util
import Super
import Super_simple
import numpy as np
import pandas as pd
from sklearn.metrics import hamming_loss, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from skmultilearn.adapt import MLkNN
from skmultilearn.model_selection import IterativeStratification
from skmultilearn.dataset import load_dataset
import time
import warnings
warnings.filterwarnings('ignore')


"""
Standard classification with full features vs Super classification with full features.
Try different number of clusters.
Hamming loss as metrics.
"""


def full_std_full_sup_hl(datasets_list, no_clses_list):
    for datasets, no_clses in zip(datasets_list, no_clses_list):
        for i in range(len(datasets)):
            X, y, feature_names, label_names = load_dataset(
                datasets[i], 'undivided')
            X = pd.DataFrame.sparse.from_spmatrix(X).to_numpy()
            y = pd.DataFrame.sparse.from_spmatrix(y).to_numpy()

            n_splits = 5
            k_fold = IterativeStratification(
                n_splits=n_splits, order=1, random_state=42)

            standard_hams = []
            super_hams = []
            to_print = ''
            fold_count = 0

            for train_idx, test_idx in k_fold.split(X, y):

                fold_count += 1
                print(
                    'Split ============================================================================', fold_count)

                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # get super classification acc(hl)
                dict_cls_ham = dict()   # no_cls:super_ham dictionary
                for no_cls in no_clses:
                    print('no_cls is ', no_cls)
                    X_train_df, X_test_df, y_train_df, y_test_df = Util.Convert_to_df(
                        X_train, X_test, y_train, y_test)
                    y_test_pred_super, y_test_super = Super.calc_preds(
                        X_train_df, y_train_df, X_test_df, y_test_df, no_cls)
                    super_ham = hamming_loss(y_test_super, y_test_pred_super)
                    dict_cls_ham[no_cls] = super_ham
                    print('super_ham: ', super_ham)
                    print('dict_cls_ham: ', dict_cls_ham)
                super_hams.append(dict_cls_ham)
                print('super_hams: ', super_hams)
                print("-------------------------------------------------------------")

                # get standard classficaition acc(hl)
                scaler_1 = StandardScaler()
                scaler_2 = MinMaxScaler()
                X_train_scaled_1 = scaler_1.fit_transform(X_train)
                X_test_scaled_1 = scaler_1.fit_transform(X_test)
                X_train_scaled_2 = scaler_2.fit_transform(X_train_scaled_1)
                X_test_scaled_2 = scaler_2.fit_transform(X_test_scaled_1)

                clf = MLkNN(k=3)
                clf.fit(X_train_scaled_2, y_train)
                y_test_pred = clf.predict(X_test_scaled_2)
                standard_ham = hamming_loss(y_test, y_test_pred)
                print('standard_ham: ', standard_ham)
                standard_hams.append(standard_ham)

                # to write the results
                to_print += '--------------Fold %d----------------\n' % fold_count
                to_print += 'Standard classification with full feature hamming loss: %.4f\n' % standard_ham
                if len(no_clses) == 2:
                    to_print += 'Super classification with full feature hamming loss with 2 clusters: %.4f\n' % dict_cls_ham[
                        2]
                    to_print += 'Super classification with full feature hamming loss with 4 clusters: %.4f\n' % dict_cls_ham[
                        4]
                elif len(no_clses) == 4:
                    to_print += 'Super classification with full feature hamming loss with 2 clusters: %.4f\n' % dict_cls_ham[
                        2]
                    to_print += 'Super classification with full feature hamming loss with 4 clusters: %.4f\n' % dict_cls_ham[
                        4]
                    to_print += 'Super classification with full feature hamming loss with 6 clusters: %.4f\n' % dict_cls_ham[
                        6]
                    to_print += 'Super classification with full feature hamming loss with 8 clusters: %.4f\n' % dict_cls_ham[
                        8]
                else:
                    to_print += 'Super classification with full feature hamming loss with 2 clusters: %.4f\n' % dict_cls_ham[
                        2]
                    to_print += 'Super classification with full feature hamming loss with 4 clusters: %.4f\n' % dict_cls_ham[
                        4]
                    to_print += 'Super classification with full feature hamming loss with 6 clusters: %.4f\n' % dict_cls_ham[
                        6]
                    to_print += 'Super classification with full feature hamming loss with 8 clusters: %.4f\n' % dict_cls_ham[
                        8]
                    to_print += 'Super classification with full feature hamming loss with 10 clusters: %.4f\n' % dict_cls_ham[
                        10]

            to_print += '--------------Average----------------\n'
            to_print += 'Ave Standard Classification Accuracy: %.4f\n' % np.average(
                standard_hams)
            # to_print += 'Ave Super Classification Accuracy: %.4f\n' % np.average(super_hams)
            if len(no_clses) == 2:
                to_print += 'Ave Super Classification with 2 clusters: %.4f\n' % np.average(
                    [super_hams[0][2], super_hams[1][2], super_hams[2][2], super_hams[3][2], super_hams[4][2]])
                to_print += 'Ave Super Classification with 4 clusters: %.4f\n' % np.average(
                    [super_hams[0][4], super_hams[1][4], super_hams[2][4], super_hams[3][4], super_hams[4][4]])
            elif len(no_clses) == 4:
                to_print += 'Ave Super Classification with 2 clusters: %.4f\n' % np.average(
                    [super_hams[0][2], super_hams[1][2], super_hams[2][2], super_hams[3][2], super_hams[4][2]])
                to_print += 'Ave Super Classification with 4 clusters: %.4f\n' % np.average(
                    [super_hams[0][4], super_hams[1][4], super_hams[2][4], super_hams[3][4], super_hams[4][4]])
                to_print += 'Ave Super Classification with 6 clusters: %.4f\n' % np.average(
                    [super_hams[0][6], super_hams[1][6], super_hams[2][6], super_hams[3][6], super_hams[4][6]])
                to_print += 'Ave Super Classification with 8 clusters: %.4f\n' % np.average(
                    [super_hams[0][8], super_hams[1][8], super_hams[2][8], super_hams[3][8], super_hams[4][8]])
            else:
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

            f = open('records/record_std_minmaxscaler_' +
                     datasets[i] + '_full_standard_super_clf.txt', 'w')
            f.write(to_print)
            f.close()


"""
Standard classification with full features vs Super classification.
With full features, with other metrics, f1_score micro, macro and weighted
"""


def full_std_full_sup_f1(datasets_list, no_clses_list):

    from sklearn.metrics import f1_score
    f1_score_total = []
    for datasets, no_clses in zip(datasets_list, no_clses_list):
        for i in range(len(datasets)):
            X, y, feature_names, label_names = load_dataset(
                datasets[i], 'undivided')
            X = pd.DataFrame.sparse.from_spmatrix(X).to_numpy()
            y = pd.DataFrame.sparse.from_spmatrix(y).to_numpy()

            n_splits = 5
            k_fold = IterativeStratification(
                n_splits=n_splits, order=1, random_state=42)

            standard_f1_micros = []
            standard_f1_macros = []
            standard_f1_weighteds = []
            super_f1_micros = []
            super_f1_macros = []
            super_f1_weighteds = []
            to_print = ''
            fold_count = 0

            for train_idx, test_idx in k_fold.split(X, y):

                fold_count += 1
                print(
                    'Split ============================================================================', fold_count)

                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # get super classification acc(f1_micro, f1_macro, f1_weighted)
                dict_cls_f1mic = dict()   # no_cls:super_ham dictionary
                dict_cls_f1mac = dict()
                dict_cls_f1wei = dict()

                for no_cls in no_clses:
                    print('no_cls is ', no_cls)
                    X_train_df, X_test_df, y_train_df, y_test_df = Util.Convert_to_df(
                        X_train, X_test, y_train, y_test)
                    y_test_pred_super, y_test_super = Super.calc_preds(
                        X_train_df, y_train_df, X_test_df, y_test_df, no_cls)
                    from sklearn.metrics import f1_score
                    super_f1mic = f1_score(
                        y_test_super, y_test_pred_super, average='micro')
                    super_f1mac = f1_score(
                        y_test_super, y_test_pred_super, average='macro')
                    super_f1wei = f1_score(
                        y_test_super, y_test_pred_super, average='weighted')

                    dict_cls_f1mic[no_cls] = super_f1mic
                    dict_cls_f1mac[no_cls] = super_f1mac
                    dict_cls_f1wei[no_cls] = super_f1wei

                    print('super_f1mic: ', super_f1mic)
                    print('dict_cls_f1mic: ', dict_cls_f1mic)
                    print('super_f1mac: ', super_f1mac)
                    print('dict_cls_f1mac: ', dict_cls_f1mac)
                    print('super_f1wei: ', super_f1wei)
                    print('dict_cls_f1wei: ', dict_cls_f1wei)

                super_f1_micros.append(dict_cls_f1mic)
                super_f1_macros.append(dict_cls_f1mac)
                super_f1_weighteds.append(dict_cls_f1wei)

                print('super_f1_micros: ', super_f1_micros)
                print('super_f1_macros: ', super_f1_macros)
                print('super_f1_weighteds', super_f1_weighteds)
                print("-------------------------------------------------------------")

                # get standard classficaition acc(hl)
                scaler_1 = StandardScaler()
                scaler_2 = MinMaxScaler()
                X_train_scaled_1 = scaler_1.fit_transform(X_train)
                X_test_scaled_1 = scaler_1.fit_transform(X_test)
                X_train_scaled_2 = scaler_2.fit_transform(X_train_scaled_1)
                X_test_scaled_2 = scaler_2.fit_transform(X_test_scaled_1)

                clf = MLkNN(k=3)
                clf.fit(X_train_scaled_2, y_train)
                y_test_pred = clf.predict(X_test_scaled_2)

                standard_f1mic = f1_score(y_test, y_test_pred, average='micro')
                standard_f1mac = f1_score(y_test, y_test_pred, average='macro')
                standard_f1wei = f1_score(
                    y_test, y_test_pred, average='weighted')

                print('standard_f1_micro: ', standard_f1mic)
                print('standard_f1_macro: ', standard_f1mac)
                print('standard_f1_weighted: ', standard_f1wei)

                standard_f1_micros.append(standard_f1mic)
                standard_f1_macros.append(standard_f1mac)
                standard_f1_weighteds.append(standard_f1wei)

                # to write the results
                to_print += '--------------Fold %d----------------\n' % fold_count
                to_print += 'Standard classification with full feature f1 micro: %.4f\n' % standard_f1mic
                to_print += 'Standard classification with full feature f1 macro: %.4f\n' % standard_f1mac
                to_print += 'Standard classification with full feature f1 weighted: %.4f\n' % standard_f1wei
                if len(no_clses) == 2:
                    to_print += 'Super classification with full feature f1 micro with 2 clusters: %.4f\n' % dict_cls_f1mic[
                        2]
                    to_print += 'Super classification with full feature f1 macro with 2 clusters: %.4f\n' % dict_cls_f1mac[
                        2]
                    to_print += 'Super classification with full feature f1 weighted with 2 clusters: %.4f\n' % dict_cls_f1wei[
                        2]
                    to_print += 'Super classification with full feature f1 micro with 4 clusters: %.4f\n' % dict_cls_f1mic[
                        4]
                    to_print += 'Super classification with full feature f1 macro with 4 clusters: %.4f\n' % dict_cls_f1mac[
                        4]
                    to_print += 'Super classification with full feature f1 weighted with 4 clusters: %.4f\n' % dict_cls_f1wei[
                        4]
                elif len(no_clses) == 4:
                    to_print += 'Super classification with full feature f1 micro with 2 clusters: %.4f\n' % dict_cls_f1mic[
                        2]
                    to_print += 'Super classification with full feature f1 macro with 2 clusters: %.4f\n' % dict_cls_f1mac[
                        2]
                    to_print += 'Super classification with full feature f1 weighted with 2 clusters: %.4f\n' % dict_cls_f1wei[
                        2]
                    to_print += 'Super classification with full feature f1 micro with 4 clusters: %.4f\n' % dict_cls_f1mic[
                        4]
                    to_print += 'Super classification with full feature f1 macro with 4 clusters: %.4f\n' % dict_cls_f1mac[
                        4]
                    to_print += 'Super classification with full feature f1 weighted with 4 clusters: %.4f\n' % dict_cls_f1wei[
                        4]
                    to_print += 'Super classification with full feature f1 micro with 6 clusters: %.4f\n' % dict_cls_f1mic[
                        6]
                    to_print += 'Super classification with full feature f1 macro with 6 clusters: %.4f\n' % dict_cls_f1mac[
                        6]
                    to_print += 'Super classification with full feature f1 weighted with 6 clusters: %.4f\n' % dict_cls_f1wei[
                        6]
                    to_print += 'Super classification with full feature f1 micro with 8 clusters: %.4f\n' % dict_cls_f1mic[
                        8]
                    to_print += 'Super classification with full feature f1 macro with 8 clusters: %.4f\n' % dict_cls_f1mac[
                        8]
                    to_print += 'Super classification with full feature f1 weighted with 8 clusters: %.4f\n' % dict_cls_f1wei[
                        8]
                else:
                    to_print += 'Super classification with full feature f1 micro with 2 clusters: %.4f\n' % dict_cls_f1mic[
                        2]
                    to_print += 'Super classification with full feature f1 macro with 2 clusters: %.4f\n' % dict_cls_f1mac[
                        2]
                    to_print += 'Super classification with full feature f1 weighted with 2 clusters: %.4f\n' % dict_cls_f1wei[
                        2]
                    to_print += 'Super classification with full feature f1 micro with 4 clusters: %.4f\n' % dict_cls_f1mic[
                        4]
                    to_print += 'Super classification with full feature f1 macro with 4 clusters: %.4f\n' % dict_cls_f1mac[
                        4]
                    to_print += 'Super classification with full feature f1 weighted with 4 clusters: %.4f\n' % dict_cls_f1wei[
                        4]
                    to_print += 'Super classification with full feature f1 micro with 6 clusters: %.4f\n' % dict_cls_f1mic[
                        6]
                    to_print += 'Super classification with full feature f1 macro with 6 clusters: %.4f\n' % dict_cls_f1mac[
                        6]
                    to_print += 'Super classification with full feature f1 weighted with 6 clusters: %.4f\n' % dict_cls_f1wei[
                        6]
                    to_print += 'Super classification with full feature f1 micro with 8 clusters: %.4f\n' % dict_cls_f1mic[
                        8]
                    to_print += 'Super classification with full feature f1 macro with 8 clusters: %.4f\n' % dict_cls_f1mac[
                        8]
                    to_print += 'Super classification with full feature f1 weighted with 8 clusters: %.4f\n' % dict_cls_f1wei[
                        8]
                    to_print += 'Super classification with full feature f1 micro with 10 clusters: %.4f\n' % dict_cls_f1mic[
                        10]
                    to_print += 'Super classification with full feature f1 macro with 10 clusters: %.4f\n' % dict_cls_f1mac[
                        10]
                    to_print += 'Super classification with full feature f1 weighted with 10 clusters: %.4f\n' % dict_cls_f1wei[
                        10]

            to_print += '--------------Average----------------\n'

            ave_std_f1mic = np.average(standard_f1_micros)
            ave_std_f1mac = np.average(standard_f1_macros)
            ave_std_f1wei = np.average(standard_f1_weighteds)
            to_print += 'Ave Standard Classification Accuracy: %.4f\n' % ave_std_f1mic
            to_print += 'Ave Standard Classification Accuracy: %.4f\n' % ave_std_f1mac
            to_print += 'Ave Standard Classification Accuracy: %.4f\n' % ave_std_f1wei

            f1_weighted.append(ave_std_f1wei)

            if len(no_clses) == 2:
                ave_2_f1mic = np.average([super_f1_micros[0][2], super_f1_micros[1][2],
                                         super_f1_micros[2][2], super_f1_micros[3][2], super_f1_micros[4][2]])
                ave_2_f1mac = np.average([super_f1_macros[0][2], super_f1_macros[1][2],
                                         super_f1_macros[2][2], super_f1_macros[3][2], super_f1_macros[4][2]])
                ave_2_f1wei = np.average([super_f1_weighteds[0][2], super_f1_weighteds[1][2],
                                         super_f1_weighteds[2][2], super_f1_weighteds[3][2], super_f1_weighteds[4][2]])
                to_print += 'Ave Super Classification with 2 clusters: %.4f\n' % ave_2_f1mic
                to_print += 'Ave Super Classification with 2 clusters: %.4f\n' % ave_2_f1mac
                to_print += 'Ave Super Classification with 2 clusters: %.4f\n' % ave_2_f1wei

                ave_4_f1mic = np.average([super_f1_micros[0][4], super_f1_micros[1][4],
                                         super_f1_micros[2][4], super_f1_micros[3][4], super_f1_micros[4][4]])
                ave_4_f1mac = np.average([super_f1_macros[0][4], super_f1_macros[1][4],
                                         super_f1_macros[2][4], super_f1_macros[3][4], super_f1_macros[4][4]])
                ave_4_f1wei = np.average([super_f1_weighteds[0][4], super_f1_weighteds[1][4],
                                         super_f1_weighteds[2][4], super_f1_weighteds[3][4], super_f1_weighteds[4][4]])
                to_print += 'Ave Super Classification with 4 clusters: %.4f\n' % ave_4_f1mic
                to_print += 'Ave Super Classification with 4 clusters: %.4f\n' % ave_4_f1mac
                to_print += 'Ave Super Classification with 4 clusters: %.4f\n' % ave_4_f1wei

                f1_micro = [ave_std_f1mic, ave_2_f1mic, ave_4_f1mic]
                f1_macro = [ave_std_f1mac, ave_2_f1mac, ave_4_f1mac]
                f1_weighted = [ave_std_f1wei, ave_2_f1wei, ave_4_f1wei]

            elif len(no_clses) == 4:
                ave_2_f1mic = np.average([super_f1_micros[0][2], super_f1_micros[1][2],
                                         super_f1_micros[2][2], super_f1_micros[3][2], super_f1_micros[4][2]])
                ave_2_f1mac = np.average([super_f1_macros[0][2], super_f1_macros[1][2],
                                         super_f1_macros[2][2], super_f1_macros[3][2], super_f1_macros[4][2]])
                ave_2_f1wei = np.average([super_f1_weighteds[0][2], super_f1_weighteds[1][2],
                                         super_f1_weighteds[2][2], super_f1_weighteds[3][2], super_f1_weighteds[4][2]])
                to_print += 'Ave Super Classification with 2 clusters: %.4f\n' % ave_2_f1mic
                to_print += 'Ave Super Classification with 2 clusters: %.4f\n' % ave_2_f1mac
                to_print += 'Ave Super Classification with 2 clusters: %.4f\n' % ave_2_f1wei

                ave_4_f1mic = np.average([super_f1_micros[0][4], super_f1_micros[1][4],
                                         super_f1_micros[2][4], super_f1_micros[3][4], super_f1_micros[4][4]])
                ave_4_f1mac = np.average([super_f1_macros[0][4], super_f1_macros[1][4],
                                         super_f1_macros[2][4], super_f1_macros[3][4], super_f1_macros[4][4]])
                ave_4_f1wei = np.average([super_f1_weighteds[0][4], super_f1_weighteds[1][4],
                                         super_f1_weighteds[2][4], super_f1_weighteds[3][4], super_f1_weighteds[4][4]])
                to_print += 'Ave Super Classification with 4 clusters: %.4f\n' % ave_4_f1mic
                to_print += 'Ave Super Classification with 4 clusters: %.4f\n' % ave_4_f1mac
                to_print += 'Ave Super Classification with 4 clusters: %.4f\n' % ave_4_f1wei

                ave_6_f1mic = np.average([super_f1_micros[0][6], super_f1_micros[1][6],
                                         super_f1_micros[2][6], super_f1_micros[3][6], super_f1_micros[4][6]])
                ave_6_f1mac = np.average([super_f1_macros[0][6], super_f1_macros[1][6],
                                         super_f1_macros[2][6], super_f1_macros[3][6], super_f1_macros[4][6]])
                ave_6_f1wei = np.average([super_f1_weighteds[0][6], super_f1_weighteds[1][6],
                                         super_f1_weighteds[2][6], super_f1_weighteds[3][6], super_f1_weighteds[4][6]])
                to_print += 'Ave Super Classification with 6 clusters: %.4f\n' % ave_6_f1mic
                to_print += 'Ave Super Classification with 6 clusters: %.4f\n' % ave_6_f1mac
                to_print += 'Ave Super Classification with 6 clusters: %.4f\n' % ave_6_f1wei

                ave_8_f1mic = np.average([super_f1_micros[0][8], super_f1_micros[1][8],
                                         super_f1_micros[2][8], super_f1_micros[3][8], super_f1_micros[4][8]])
                ave_8_f1mac = np.average([super_f1_macros[0][8], super_f1_macros[1][8],
                                         super_f1_macros[2][8], super_f1_macros[3][8], super_f1_macros[4][8]])
                ave_8_f1wei = np.average([super_f1_weighteds[0][8], super_f1_weighteds[1][8],
                                         super_f1_weighteds[2][8], super_f1_weighteds[3][8], super_f1_weighteds[4][8]])
                to_print += 'Ave Super Classification with 8 clusters: %.4f\n' % ave_8_f1mic
                to_print += 'Ave Super Classification with 8 clusters: %.4f\n' % ave_8_f1mac
                to_print += 'Ave Super Classification with 8 clusters: %.4f\n' % ave_8_f1wei

                f1_micro = [ave_std_f1mic, ave_2_f1mic,
                            ave_4_f1mic, ave_6_f1mic, ave_8_f1mic]
                f1_macro = [ave_std_f1mac, ave_2_f1mac,
                            ave_4_f1mac, ave_6_f1mac, ave_8_f1mac]
                f1_weighted = [ave_std_f1wei, ave_2_f1wei,
                               ave_4_f1wei, ave_6_f1wei, ave_8_f1wei]

            else:
                ave_2_f1mic = np.average([super_f1_micros[0][2], super_f1_micros[1][2],
                                         super_f1_micros[2][2], super_f1_micros[3][2], super_f1_micros[4][2]])
                ave_2_f1mac = np.average([super_f1_macros[0][2], super_f1_macros[1][2],
                                         super_f1_macros[2][2], super_f1_macros[3][2], super_f1_macros[4][2]])
                ave_2_f1wei = np.average([super_f1_weighteds[0][2], super_f1_weighteds[1][2],
                                         super_f1_weighteds[2][2], super_f1_weighteds[3][2], super_f1_weighteds[4][2]])
                to_print += 'Ave Super Classification with 2 clusters: %.4f\n' % ave_2_f1mic
                to_print += 'Ave Super Classification with 2 clusters: %.4f\n' % ave_2_f1mac
                to_print += 'Ave Super Classification with 2 clusters: %.4f\n' % ave_2_f1wei

                ave_4_f1mic = np.average([super_f1_micros[0][4], super_f1_micros[1][4],
                                         super_f1_micros[2][4], super_f1_micros[3][4], super_f1_micros[4][4]])
                ave_4_f1mac = np.average([super_f1_macros[0][4], super_f1_macros[1][4],
                                         super_f1_macros[2][4], super_f1_macros[3][4], super_f1_macros[4][4]])
                ave_4_f1wei = np.average([super_f1_weighteds[0][4], super_f1_weighteds[1][4],
                                         super_f1_weighteds[2][4], super_f1_weighteds[3][4], super_f1_weighteds[4][4]])
                to_print += 'Ave Super Classification with 4 clusters: %.4f\n' % ave_4_f1mic
                to_print += 'Ave Super Classification with 4 clusters: %.4f\n' % ave_4_f1mac
                to_print += 'Ave Super Classification with 4 clusters: %.4f\n' % ave_4_f1wei

                ave_6_f1mic = np.average([super_f1_micros[0][6], super_f1_micros[1][6],
                                         super_f1_micros[2][6], super_f1_micros[3][6], super_f1_micros[4][6]])
                ave_6_f1mac = np.average([super_f1_macros[0][6], super_f1_macros[1][6],
                                         super_f1_macros[2][6], super_f1_macros[3][6], super_f1_macros[4][6]])
                ave_6_f1wei = np.average([super_f1_weighteds[0][6], super_f1_weighteds[1][6],
                                         super_f1_weighteds[2][6], super_f1_weighteds[3][6], super_f1_weighteds[4][6]])
                to_print += 'Ave Super Classification with 6 clusters: %.4f\n' % ave_6_f1mic
                to_print += 'Ave Super Classification with 6 clusters: %.4f\n' % ave_6_f1mac
                to_print += 'Ave Super Classification with 6 clusters: %.4f\n' % ave_6_f1wei

                ave_8_f1mic = np.average([super_f1_micros[0][8], super_f1_micros[1][8],
                                         super_f1_micros[2][8], super_f1_micros[3][8], super_f1_micros[4][8]])
                ave_8_f1mac = np.average([super_f1_macros[0][8], super_f1_macros[1][8],
                                         super_f1_macros[2][8], super_f1_macros[3][8], super_f1_macros[4][8]])
                ave_8_f1wei = np.average([super_f1_weighteds[0][8], super_f1_weighteds[1][8],
                                         super_f1_weighteds[2][8], super_f1_weighteds[3][8], super_f1_weighteds[4][8]])
                to_print += 'Ave Super Classification with 8 clusters: %.4f\n' % ave_8_f1mic
                to_print += 'Ave Super Classification with 8 clusters: %.4f\n' % ave_8_f1mac
                to_print += 'Ave Super Classification with 8 clusters: %.4f\n' % ave_8_f1wei

                ave_10_f1mic = np.average([super_f1_micros[0][10], super_f1_micros[1][10],
                                          super_f1_micros[2][10], super_f1_micros[3][10], super_f1_micros[4][10]])
                ave_10_f1mac = np.average([super_f1_macros[0][10], super_f1_macros[1][10],
                                          super_f1_macros[2][10], super_f1_macros[3][10], super_f1_macros[4][10]])
                ave_10_f1wei = np.average([super_f1_weighteds[0][10], super_f1_weighteds[1][10],
                                          super_f1_weighteds[2][10], super_f1_weighteds[3][10], super_f1_weighteds[4][10]])
                to_print += 'Ave Super Classification with 10 clusters: %.4f\n' % ave_10_f1mic
                to_print += 'Ave Super Classification with 10 clusters: %.4f\n' % ave_10_f1mac
                to_print += 'Ave Super Classification with 10 clusters: %.4f\n' % ave_10_f1wei

                f1_micro = [ave_std_f1mic, ave_2_f1mic, ave_4_f1mic,
                            ave_6_f1mic, ave_8_f1mic, ave_10_f1mic]
                f1_macro = [ave_std_f1mac, ave_2_f1mac, ave_4_f1mac,
                            ave_6_f1mac, ave_8_f1mac, ave_10_f1mac]
                f1_weighted = [ave_std_f1wei, ave_2_f1wei,
                               ave_4_f1wei, ave_6_f1wei, ave_8_f1wei, ave_10_f1wei]

            f1_score = [f1_micro, f1_macro, f1_weighted]
            dict_f1 = dict()
            # dictionary datasetname : [f1_mic, f1_mac, f1_wei]
            dict_f1[datasets[i]] = f1_score
            f1_score_total.append(dict_f1)

            f = open('records/f1_score/record_f1_mic_mac_wei' +
                     datasets[i] + '_full_standard_super_clf.txt', 'w')
            f.write(to_print)
            f.close()

    """
    Full features, standard classification vs super PSO selected features
    F1 score micro as metrics.
    """


def full_std_sel_sup_f1(datasets_list, no_clses_list):
    from sklearn.metrics import f1_score
    for datasets, no_cls in zip(datasets_list, no_clses_list):
        for i in range(len(datasets)):
            print(
                datasets[i] + " is running: ===================================================")
            X, y, feature_names, label_names = load_dataset(
                datasets[i], 'undivided')
            n_features = len(feature_names)
            X = pd.DataFrame.sparse.from_spmatrix(X).to_numpy()
            y = pd.DataFrame.sparse.from_spmatrix(y).to_numpy()

            n_splits = 5
            k_fold = IterativeStratification(
                n_splits=n_splits, order=1, random_state=42)

            full_f1_mics = []
            sel_f1_mics = []
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
                # get standard classficaition acc(f1)
                scaler_1 = StandardScaler()
                scaler_2 = MinMaxScaler()
                X_train_scaled_1 = scaler_1.fit_transform(X_train)
                X_test_scaled_1 = scaler_1.fit_transform(X_test)
                X_train_scaled_2 = scaler_2.fit_transform(X_train_scaled_1)
                X_test_scaled_2 = scaler_2.fit_transform(X_test_scaled_1)

                clf = MLkNN(k=3)
                clf.fit(X_train_scaled_2, y_train)
                y_test_pred = clf.predict(X_test_scaled_2)

                full_f1_mic = f1_score(y_test, y_test_pred, average='micro')
                print('Full_f1_micro: ', full_f1_mic)
                full_f1_mics.append(full_f1_mic)

                #  perform FS
                start_PSO = time.time_ns()    # marking start time of PSO
                problem = Problem.FS_ML_super_f1(
                    minimize=False, X=X_train, y=y_train, no_cls=no_cls)

                # parameter for PSO
                pop_size = 100
                n_iterations = 100
                swarm = Swarm.Swarm(n_particles=pop_size, length=n_features, pos_max=1.0, pos_min=0,
                                    vel_max=0.2, vel_min=-0.2, problem=problem, n_iterations=n_iterations)
                best_sol, best_fit = swarm.iterate()
                end_PSO = time.time_ns()      # marking ending time of PSO
                duration_PSO = round((end_PSO - start_PSO)/1000000000, 2)
                PSO_durations.append(duration_PSO)

                # process the final solution
                sel_fea = np.where(best_sol > problem.threshold)[0]
                clf.fit(X_train_scaled_2[:, sel_fea], y_train)
                y_test_pred = clf.predict(X_test_scaled_2[:, sel_fea])
                sel_f1_mic = f1_score(y_test, y_test_pred, average='micro')
                print('Selected_f1_micro: ', sel_f1_mic)
                sel_f1_mics.append(sel_f1_mic)
                f_ratios.append(len(sel_fea)/n_features)

                # to write the results
                to_print += '--------------Fold %d----------------\n' % fold_count
                to_print += 'Full feature f1 score micro: %.4f\n' % full_f1_mic
                to_print += 'Fold selected f1 score micro: %.4f\n' % sel_f1_mic
                to_print += 'Time of PSO: %.4f\n' % duration_PSO
                to_print += 'Selection ratio: %.2f\n' % (
                    len(sel_fea)/n_features)
                to_print += 'Selected features: %s\n' % (
                    ', '.join([str(ele) for ele in sel_fea]))

            to_print += '--------------Average----------------\n'
            to_print += 'Ave Full Accuracy: %.4f\n' % np.average(full_f1_mics)
            to_print += 'Ave Selection Accuracy: %.4f\n' % np.average(
                sel_f1_mics)
            to_print += 'Ave time of PSO: %.4f\n' % np.average(PSO_durations)
            to_print += 'Ave Feature Ratio: %.2f\n' % np.average(f_ratios)

            f = open('records/f1_score_sup_PSO_more/record_' +
                     datasets[i] + 'std_full_superPSO_selected.txt', 'w')
            f.write(to_print)
            f.close()


"""
Full features, standard classification vs standard PSO selected features
F1 score micro as metrics.
"""


def full_std_sel_std_f1(datasets_list):
    from sklearn.metrics import f1_score
    for datasets in datasets_list:
        for i in range(len(datasets)):
            print(
                datasets[i] + " is running: ===================================================")
            X, y, feature_names, label_names = load_dataset(
                datasets[i], 'undivided')
            n_features = len(feature_names)
            X = pd.DataFrame.sparse.from_spmatrix(X).to_numpy()
            y = pd.DataFrame.sparse.from_spmatrix(y).to_numpy()

            n_splits = 5
            k_fold = IterativeStratification(
                n_splits=n_splits, order=1, random_state=42)

            full_f1_mics = []
            sel_f1_mics = []
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
                # get standard classficaition acc(f1)
                scaler_1 = StandardScaler()
                scaler_2 = MinMaxScaler()
                X_train_scaled_1 = scaler_1.fit_transform(X_train)
                X_test_scaled_1 = scaler_1.fit_transform(X_test)
                X_train_scaled_2 = scaler_2.fit_transform(X_train_scaled_1)
                X_test_scaled_2 = scaler_2.fit_transform(X_test_scaled_1)

                clf = MLkNN(k=3)
                clf.fit(X_train_scaled_2, y_train)
                y_test_pred = clf.predict(X_test_scaled_2)

                full_f1_mic = f1_score(y_test, y_test_pred, average='micro')
                print('Full_f1_micro: ', full_f1_mic)
                full_f1_mics.append(full_f1_mic)

                #  perform FS
                start_PSO = time.time_ns()    # marking start time of PSO
                problem = Problem.FS_ML_f1(
                    minimize=False, X=X_train, y=y_train)

                # parameter for PSO
                pop_size = 100
                n_iterations = 100
                swarm = Swarm.Swarm(n_particles=pop_size, length=n_features, pos_max=1.0, pos_min=0,
                                    vel_max=0.2, vel_min=-0.2, problem=problem, n_iterations=n_iterations)
                best_sol, best_fit = swarm.iterate()
                end_PSO = time.time_ns()      # marking ending time of PSO
                duration_PSO = round((end_PSO - start_PSO)/1000000000, 2)
                PSO_durations.append(duration_PSO)

                # process the final solution
                sel_fea = np.where(best_sol > problem.threshold)[0]
                clf.fit(X_train_scaled_2[:, sel_fea], y_train)
                y_test_pred = clf.predict(X_test_scaled_2[:, sel_fea])
                sel_f1_mic = f1_score(y_test, y_test_pred, average='micro')
                print('Selected_f1_micro: ', sel_f1_mic)
                sel_f1_mics.append(sel_f1_mic)
                f_ratios.append(len(sel_fea)/n_features)

                # to write the results
                to_print += '--------------Fold %d----------------\n' % fold_count
                to_print += 'Full feature f1 score micro: %.4f\n' % full_f1_mic
                to_print += 'Fold selected f1 score micro: %.4f\n' % sel_f1_mic
                to_print += 'Time of PSO: %.4f\n' % duration_PSO
                to_print += 'Selection ratio: %.2f\n' % (
                    len(sel_fea)/n_features)
                to_print += 'Selected features: %s\n' % (
                    ', '.join([str(ele) for ele in sel_fea]))

            to_print += '--------------Average----------------\n'
            to_print += 'Ave Full Accuracy: %.4f\n' % np.average(full_f1_mics)
            to_print += 'Ave Selection Accuracy: %.4f\n' % np.average(
                sel_f1_mics)
            to_print += 'Ave time of PSO: %.4f\n' % np.average(PSO_durations)
            to_print += 'Ave Feature Ratio: %.2f\n' % np.average(f_ratios)

            f = open('records/f1_score_std_PSO_more/record_' +
                     datasets[i] + 'std_full_stdPSO_selected.txt', 'w')
            f.write(to_print)
            f.close()


"""
    Full features, standard classification vs super PSO selected features
    Hamming loss as metrics.
    """


def full_std_sel_sup_hl(datasets_list, no_clses_list):
    for datasets, no_cls in zip(datasets_list, no_clses_list):
        for i in range(len(datasets)):
            print(
                datasets[i] + " is running: ===================================================")
            X, y, feature_names, label_names = load_dataset(
                datasets[i], 'undivided')
            n_features = len(feature_names)
            X = pd.DataFrame.sparse.from_spmatrix(X).to_numpy()
            y = pd.DataFrame.sparse.from_spmatrix(y).to_numpy()

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
                # get standard classficaition acc(f1)
                scaler_1 = StandardScaler()
                scaler_2 = MinMaxScaler()
                X_train_scaled_1 = scaler_1.fit_transform(X_train)
                X_test_scaled_1 = scaler_1.fit_transform(X_test)
                X_train_scaled_2 = scaler_2.fit_transform(X_train_scaled_1)
                X_test_scaled_2 = scaler_2.fit_transform(X_test_scaled_1)

                clf = MLkNN(k=3)
                clf.fit(X_train_scaled_2, y_train)
                y_test_pred = clf.predict(X_test_scaled_2)

                full_ham = hamming_loss(y_test, y_test_pred)
                print('Full_hamming_loss: ', full_ham)
                full_hams.append(full_ham)

                #  perform FS
                start_PSO = time.time_ns()    # marking start time of PSO
                problem = Problem.FS_ML_super_hl(
                    minimize=True, X=X_train, y=y_train, no_cls=no_cls)

                # parameter for PSO
                pop_size = 100
                n_iterations = 100
                swarm = Swarm.Swarm(n_particles=pop_size, length=n_features, pos_max=1.0, pos_min=0,
                                    vel_max=0.2, vel_min=-0.2, problem=problem, n_iterations=n_iterations)
                best_sol, best_fit = swarm.iterate()
                end_PSO = time.time_ns()      # marking ending time of PSO
                duration_PSO = round((end_PSO - start_PSO)/1000000000, 2)
                PSO_durations.append(duration_PSO)

                # process the final solution
                sel_fea = np.where(best_sol > problem.threshold)[0]
                clf.fit(X_train_scaled_2[:, sel_fea], y_train)
                y_test_pred = clf.predict(X_test_scaled_2[:, sel_fea])
                sel_ham = hamming_loss(y_test, y_test_pred)
                print('Selected_hamming_loss: ', sel_ham)
                sel_hams.append(sel_ham)
                f_ratios.append(len(sel_fea)/n_features)

                # to write the results
                to_print += '--------------Fold %d----------------\n' % fold_count
                to_print += 'Full feature f1 score micro: %.4f\n' % full_ham
                to_print += 'Fold selected f1 score micro: %.4f\n' % sel_ham
                to_print += 'Time of PSO: %.4f\n' % duration_PSO
                to_print += 'Selection ratio: %.2f\n' % (
                    len(sel_fea)/n_features)
                to_print += 'Selected features: %s\n' % (
                    ', '.join([str(ele) for ele in sel_fea]))

            to_print += '--------------Average----------------\n'
            to_print += 'Ave Full Accuracy: %.4f\n' % np.average(full_hams)
            to_print += 'Ave Selection Accuracy: %.4f\n' % np.average(
                sel_hams)
            to_print += 'Ave time of PSO: %.4f\n' % np.average(PSO_durations)
            to_print += 'Ave Feature Ratio: %.2f\n' % np.average(f_ratios)

            f = open('records/hamming_loss_sup_PSO_more/record_' +
                     datasets[i] + '_std_full_superPSO_selected.txt', 'w')
            f.write(to_print)
            f.close()


"""
Full features, standard classification vs standard PSO selected features
F1 score micro as metrics.
"""


def full_std_sel_std_hl(datasets_list):
    for datasets in datasets_list:
        for i in range(len(datasets)):
            print(
                datasets[i] + " is running: ===================================================")
            X, y, feature_names, label_names = load_dataset(
                datasets[i], 'undivided')
            n_features = len(feature_names)
            X = pd.DataFrame.sparse.from_spmatrix(X).to_numpy()
            y = pd.DataFrame.sparse.from_spmatrix(y).to_numpy()

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
                # get standard classficaition acc(f1)
                scaler_1 = StandardScaler()
                scaler_2 = MinMaxScaler()
                X_train_scaled_1 = scaler_1.fit_transform(X_train)
                X_test_scaled_1 = scaler_1.fit_transform(X_test)
                X_train_scaled_2 = scaler_2.fit_transform(X_train_scaled_1)
                X_test_scaled_2 = scaler_2.fit_transform(X_test_scaled_1)

                clf = MLkNN(k=3)
                clf.fit(X_train_scaled_2, y_train)
                y_test_pred = clf.predict(X_test_scaled_2)

                full_ham = hamming_loss(y_test, y_test_pred)
                print('Full_hamming_loss: ', full_ham)
                full_hams.append(full_ham)

                #  perform FS
                start_PSO = time.time_ns()    # marking start time of PSO
                problem = Problem.FS_ML_hl(
                    minimize=True, X=X_train, y=y_train)

                # parameter for PSO
                pop_size = 100
                n_iterations = 100
                swarm = Swarm.Swarm(n_particles=pop_size, length=n_features, pos_max=1.0, pos_min=0,
                                    vel_max=0.2, vel_min=-0.2, problem=problem, n_iterations=n_iterations)
                best_sol, best_fit = swarm.iterate()
                end_PSO = time.time_ns()      # marking ending time of PSO
                duration_PSO = round((end_PSO - start_PSO)/1000000000, 2)
                PSO_durations.append(duration_PSO)

                # process the final solution
                sel_fea = np.where(best_sol > problem.threshold)[0]
                clf.fit(X_train_scaled_2[:, sel_fea], y_train)
                y_test_pred = clf.predict(X_test_scaled_2[:, sel_fea])
                sel_ham = hamming_loss(y_test, y_test_pred)
                print('Selected_hamming_loss: ', sel_ham)
                sel_hams.append(sel_ham)
                f_ratios.append(len(sel_fea)/n_features)

                # to write the results
                to_print += '--------------Fold %d----------------\n' % fold_count
                to_print += 'Full feature f1 score micro: %.4f\n' % full_ham
                to_print += 'Fold selected f1 score micro: %.4f\n' % sel_ham
                to_print += 'Time of PSO: %.4f\n' % duration_PSO
                to_print += 'Selection ratio: %.2f\n' % (
                    len(sel_fea)/n_features)
                to_print += 'Selected features: %s\n' % (
                    ', '.join([str(ele) for ele in sel_fea]))

            to_print += '--------------Average----------------\n'
            to_print += 'Ave Full Accuracy: %.4f\n' % np.average(full_hams)
            to_print += 'Ave Selection Accuracy: %.4f\n' % np.average(
                sel_hams)
            to_print += 'Ave time of PSO: %.4f\n' % np.average(PSO_durations)
            to_print += 'Ave Feature Ratio: %.2f\n' % np.average(f_ratios)

            f = open('records/hamming_loss_std_PSO_more/record_' +
                     datasets[i] + '_std_full_stdPSO_selected.txt', 'w')
            f.write(to_print)
            f.close()


"""
Put std and sup PSO together. 
No random_state.
Running on grid.
"""


def full_std_PSOsel_std_sup_f1(datasets_list, no_clses_list, run):
    from sklearn.metrics import f1_score
    for datasets, no_cls in zip(datasets_list, no_clses_list):
        for i in range(len(datasets)):
            print(
                datasets[i] + " is running: ===================================================")
            X, y, feature_names, label_names = load_dataset(
                datasets[i], 'undivided')
            n_features = len(feature_names)
            X = pd.DataFrame.sparse.from_spmatrix(X).to_numpy()
            y = pd.DataFrame.sparse.from_spmatrix(y).to_numpy()

            n_splits = 3
            k_fold = IterativeStratification(
                n_splits=n_splits, order=1, random_state=42)

            full_f1_mics = []
            sel_std_f1_mics = []
            sel_sup_f1_mics = []
            PSO_std_durations = []
            PSO_sup_durations = []
            f_std_ratios = []
            f_sup_ratios = []
            to_print = ''
            fold_count = 0

            for train_idx, test_idx in k_fold.split(X, y):
                print('Fold ', fold_count)
                fold_count += 1

                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # get full acc
                # get standard classficaition acc(f1)
                scaler_1 = StandardScaler()
                scaler_2 = MinMaxScaler()
                X_train_scaled_1 = scaler_1.fit_transform(X_train)
                X_test_scaled_1 = scaler_1.fit_transform(X_test)
                X_train_scaled_2 = scaler_2.fit_transform(X_train_scaled_1)
                X_test_scaled_2 = scaler_2.fit_transform(X_test_scaled_1)

                clf = MLkNN(k=3)
                clf.fit(X_train_scaled_2, y_train)
                y_test_pred = clf.predict(X_test_scaled_2)

                full_f1_mic = f1_score(y_test, y_test_pred, average='micro')
                print('Full_f1_micro: ', full_f1_mic)
                full_f1_mics.append(full_f1_mic)

                #  perform standard PSO FS
                start_std_PSO = time.time_ns()    # marking start time of PSO
                problem = Problem.FS_ML_f1(
                    minimize=False, X=X_train, y=y_train)

                # parameter for standard PSO
                pop_size = 3
                n_iterations = 5
                swarm = Swarm.Swarm(n_particles=pop_size, length=n_features, pos_max=1.0, pos_min=0,
                                    vel_max=0.2, vel_min=-0.2, problem=problem, n_iterations=n_iterations)
                best_sol_std, best_fit_std = swarm.iterate()
                end_std_PSO = time.time_ns()      # marking ending time of PSO
                duration_std_PSO = round(
                    (end_std_PSO - start_std_PSO)/1000000000, 2)
                PSO_std_durations.append(duration_std_PSO)

                # process the final solution
                sel_fea_std = np.where(best_sol_std > problem.threshold)[0]
                clf.fit(X_train_scaled_2[:, sel_fea_std], y_train)
                y_test_pred = clf.predict(X_test_scaled_2[:, sel_fea_std])
                sel_std_f1_mic = f1_score(y_test, y_test_pred, average='micro')
                print('Selected_standard_f1_micro: ', sel_std_f1_mic)
                sel_std_f1_mics.append(sel_std_f1_mic)
                f_std_ratios.append(len(sel_fea_std)/n_features)

                #  perform super PSO FS
                start_sup_PSO = time.time_ns()    # marking start time of PSO
                problem = Problem.FS_ML_super_f1(
                    minimize=False, X=X_train, y=y_train, no_cls=no_cls)

                # parameter for PSO
                pop_size = 3
                n_iterations = 5
                swarm = Swarm.Swarm(n_particles=pop_size, length=n_features, pos_max=1.0, pos_min=0,
                                    vel_max=0.2, vel_min=-0.2, problem=problem, n_iterations=n_iterations)
                best_sol_sup, best_fit_sup = swarm.iterate()
                end_sup_PSO = time.time_ns()      # marking ending time of PSO
                duration_sup_PSO = round(
                    (end_sup_PSO - start_sup_PSO)/1000000000, 2)
                PSO_sup_durations.append(duration_sup_PSO)

                # process the final solution
                sel_fea_sup = np.where(best_sol_sup > problem.threshold)[0]
                clf.fit(X_train_scaled_2[:, sel_fea_sup], y_train)
                y_test_pred = clf.predict(X_test_scaled_2[:, sel_fea_sup])
                sel_sup_f1_mic = f1_score(y_test, y_test_pred, average='micro')
                print('Selected_super_f1_micro: ', sel_sup_f1_mic)
                sel_sup_f1_mics.append(sel_sup_f1_mic)
                f_sup_ratios.append(len(sel_fea_sup)/n_features)

                # to write the results
                to_print += '--------------Fold %d----------------\n' % fold_count
                to_print += 'Full feature f1 score micro: %.4f\n' % full_f1_mic
                to_print += 'Fold selected standard PSO f1 score micro: %.4f\n' % sel_std_f1_mic
                to_print += 'Fold selected super PSO f1 score micro: %.4f\n' % sel_sup_f1_mic
                to_print += 'Time of standard PSO: %.4f\n' % duration_std_PSO
                to_print += 'Time of super PSO: %.4f\n' % duration_sup_PSO
                to_print += 'Selection standard PSO ratio: %.2f\n' % (
                    len(sel_fea_std)/n_features)
                to_print += 'Selection super PSO ratio: %.2f\n' % (
                    len(sel_fea_sup)/n_features)
                to_print += 'Selected features standard PSO: %s\n' % (
                    ', '.join([str(ele) for ele in sel_fea_std]))
                to_print += 'Selected features super PSO: %s\n' % (
                    ', '.join([str(ele) for ele in sel_fea_sup]))

            to_print += '--------------Average----------------\n'
            to_print += 'Ave Full Accuracy: %.4f\n' % np.average(full_f1_mics)
            to_print += 'Ave Standard PSO Selection Accuracy: %.4f\n' % np.average(
                sel_std_f1_mics)
            to_print += 'Ave Super PSO Selection Accuracy: %.4f\n' % np.average(
                sel_sup_f1_mics)
            to_print += 'Ave time of Standard PSO: %.4f\n' % np.average(
                PSO_std_durations)
            to_print += 'Ave time of Super PSO: %.4f\n' % np.average(
                PSO_sup_durations)
            to_print += 'Ave Standard PSO Feature Ratio: %.2f\n' % np.average(
                f_std_ratios)
            to_print += 'Ave Super PSO Feature Ratio: %.2f\n' % np.average(
                f_sup_ratios)

            # f = open(datasets[i] + str(run) + '_std_full_superPSO_selected.txt', 'w')
            f = open('records/f1_score_full_std_sup_PSO_comparison/record_' +
                     datasets[i] + str(run) + 'std_full_superPSO_selected.txt', 'w')
            f.write(to_print)
            f.close()


def fullstd_PSOsel_std_sup_simple_f1(datasets_list, no_clses_list, run):
    from sklearn.metrics import f1_score
    for datasets, no_cls in zip(datasets_list, no_clses_list):
        for i in range(len(datasets)):
            X, y, feature_names, label_names = load_dataset(
                datasets[i], 'undivided')
            n_features = len(feature_names)
            X = pd.DataFrame.sparse.from_spmatrix(X).to_numpy()
            y = pd.DataFrame.sparse.from_spmatrix(y).to_numpy()

            n_splits = 3
            k_fold = IterativeStratification(
                n_splits=n_splits, order=1)

            full_f1_mics = []
            full_f1_macs = []

            sel_std_f1_mics = []
            sel_std_f1_macs = []
            sel_sup_f1_mics = []
            sel_sup_f1_macs = []
            sel_sim_f1_mics = []
            sel_sim_f1_macs = []

            PSO_std_durations = []
            PSO_sup_durations = []
            PSO_sim_durations = []

            f_std_ratios = []
            f_sup_ratios = []
            f_sim_ratios = []
            to_print = ''
            fold_count = 0

            for train_idx, test_idx in k_fold.split(X, y):
                print('Fold ', fold_count)
                fold_count += 1

                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # get full acc
                # get standard classficaition acc(f1)
                scaler_1 = StandardScaler()
                scaler_2 = MinMaxScaler()
                X_train_scaled_1 = scaler_1.fit_transform(X_train)
                X_test_scaled_1 = scaler_1.fit_transform(X_test)
                X_train_scaled_2 = scaler_2.fit_transform(X_train_scaled_1)
                X_test_scaled_2 = scaler_2.fit_transform(X_test_scaled_1)

                clf = MLkNN(k=3)
                clf.fit(X_train_scaled_2, y_train)
                y_test_pred = clf.predict(X_test_scaled_2)

                full_f1_mic = f1_score(y_test, y_test_pred, average='micro')
                print('Full_f1_micro: ', full_f1_mic)
                full_f1_mics.append(full_f1_mic)

                full_f1_mac = f1_score(y_test, y_test_pred, average='macro')
                print('Full_f1_macro: ', full_f1_mac)
                full_f1_macs.append(full_f1_mac)

                #  perform standard PSO FS
                start_std_PSO = time.time_ns()    # marking start time of PSO
                problem_std = Problem.FS_ML_f1(
                    minimize=False, X=X_train, y=y_train)

                pop_size = 3
                n_iterations = 5
                swarm = Swarm.Swarm(n_particles=pop_size, length=n_features, pos_max=1.0, pos_min=0,
                                    vel_max=0.2, vel_min=-0.2, problem=problem_std, n_iterations=n_iterations)
                best_sol_std, best_fit_std = swarm.iterate()
                end_std_PSO = time.time_ns()      # marking ending time of PSO
                duration_std_PSO = round(
                    (end_std_PSO - start_std_PSO)/1000000000, 2)
                PSO_std_durations.append(duration_std_PSO)

                sel_fea_std = np.where(best_sol_std > problem_std.threshold)[0]
                clf.fit(X_train_scaled_2[:, sel_fea_std], y_train)
                y_test_pred = clf.predict(X_test_scaled_2[:, sel_fea_std])
                sel_std_f1_mic = f1_score(y_test, y_test_pred, average='micro')
                print('Selected_standard_f1_micro: ', sel_std_f1_mic)
                sel_std_f1_mics.append(sel_std_f1_mic)
                sel_std_f1_mac = f1_score(y_test, y_test_pred, average='macro')
                print('Selected_standard_f1_macro: ', sel_std_f1_mac)
                sel_std_f1_macs.append(sel_std_f1_mac)
                f_std_ratios.append(len(sel_fea_std)/n_features)

                #  perform super PSO FS
                start_sup_PSO = time.time_ns()    # marking start time of PSO
                problem_sup = Problem.FS_ML_super_f1(
                    minimize=False, X=X_train, y=y_train, no_cls=no_cls)

                pop_size = 3
                n_iterations = 5
                swarm = Swarm.Swarm(n_particles=pop_size, length=n_features, pos_max=1.0, pos_min=0,
                                    vel_max=0.2, vel_min=-0.2, problem=problem_sup, n_iterations=n_iterations)
                best_sol_sup, best_fit_sup = swarm.iterate()
                end_sup_PSO = time.time_ns()      # marking ending time of PSO
                duration_sup_PSO = round(
                    (end_sup_PSO - start_sup_PSO)/1000000000, 2)
                PSO_sup_durations.append(duration_sup_PSO)

                sel_fea_sup = np.where(best_sol_sup > problem_sup.threshold)[0]
                clf.fit(X_train_scaled_2[:, sel_fea_sup], y_train)
                y_test_pred = clf.predict(X_test_scaled_2[:, sel_fea_sup])
                sel_sup_f1_mic = f1_score(y_test, y_test_pred, average='micro')
                print('Selected_super_f1_micro: ', sel_sup_f1_mic)
                sel_sup_f1_mics.append(sel_sup_f1_mic)
                sel_sup_f1_mac = f1_score(y_test, y_test_pred, average='macro')
                print('Selected_super_f1_macro: ', sel_sup_f1_mac)
                sel_sup_f1_macs.append(sel_sup_f1_mac)
                f_sup_ratios.append(len(sel_fea_sup)/n_features)

                # perform super simple PSO FS
                start_sim_PSO = time.time_ns()    # marking start time of PSO

                y_train_s = Super_simple.label_convert_simple(
                    y_train, no_cls)   # Convert y_train
                problem_sim = Problem.FS_ML_super_simple_f1(
                    minimize=False, X=X_train, y=y_train_s, no_cls=no_cls)

                pop_size = 3
                n_iterations = 5
                swarm = Swarm.Swarm(n_particles=pop_size, length=n_features, pos_max=1.0, pos_min=0,
                                    vel_max=0.2, vel_min=-0.2, problem=problem_sim, n_iterations=n_iterations)
                best_sol_sim, best_fit_sim = swarm.iterate()
                end_sim_PSO = time.time_ns()      # marking ending time of PSO
                duration_sim_PSO = round(
                    (end_sim_PSO - start_sim_PSO)/1000000000, 2)
                PSO_sim_durations.append(duration_sim_PSO)

                sel_fea_sim = np.where(best_sol_sim > problem_sim.threshold)[0]
                clf.fit(X_train_scaled_2[:, sel_fea_sim], y_train)
                y_test_pred = clf.predict(X_test_scaled_2[:, sel_fea_sim])
                sel_sim_f1_mic = f1_score(y_test, y_test_pred, average='micro')
                print('Selected_super_simple_f1_micro: ', sel_sim_f1_mic)
                sel_sim_f1_mics.append(sel_sim_f1_mic)
                sel_sim_f1_mac = f1_score(y_test, y_test_pred, average='macro')
                print('Selected_super_simple_f1_macro: ', sel_sim_f1_mac)
                sel_sim_f1_macs.append(sel_sim_f1_mac)
                f_sim_ratios.append(len(sel_fea_sim)/n_features)

                # to write the results
                to_print += '--------------Fold %d----------------\n' % fold_count
                to_print += 'Full feature f1 score micro: %.4f\n' % full_f1_mic
                to_print += 'Fold selected standard PSO f1 score micro: %.4f\n' % sel_std_f1_mic
                to_print += 'Fold selected super PSO f1 score micro: %.4f\n' % sel_sup_f1_mic
                to_print += 'Fold selected super simple PSO f1 score micro: %.4f\n' % sel_sim_f1_mic

                to_print += 'Full feature f1 score macro: %.4f\n' % full_f1_mac
                to_print += 'Fold selected standard PSO f1 score macro: %.4f\n' % sel_std_f1_mac
                to_print += 'Fold selected super PSO f1 score macro: %.4f\n' % sel_sup_f1_mac
                to_print += 'Fold selected super simple PSO f1 score macro: %.4f\n' % sel_sim_f1_mac

                to_print += 'Time of standard PSO: %.4f\n' % duration_std_PSO
                to_print += 'Time of super PSO: %.4f\n' % duration_sup_PSO
                to_print += 'Time of super simple PSO: %.4f\n' % duration_sim_PSO
                to_print += 'Selection standard PSO ratio: %.2f\n' % (
                    len(sel_fea_std)/n_features)
                to_print += 'Selection super PSO ratio: %.2f\n' % (
                    len(sel_fea_sup)/n_features)
                to_print += 'Selection super simple PSO ratio: %.2f\n' % (
                    len(sel_fea_sim)/n_features)
                to_print += 'Selected features standard PSO: %s\n' % (
                    ', '.join([str(ele) for ele in sel_fea_std]))
                to_print += 'Selected features super PSO: %s\n' % (
                    ', '.join([str(ele) for ele in sel_fea_sup]))
                to_print += 'Selected features super simple PSO: %s\n' % (
                    ', '.join([str(ele) for ele in sel_fea_sim]))

            to_print += '--------------Average----------------\n'
            to_print += 'Ave Full Accuracy micro: %.4f\n' % np.average(
                full_f1_mics)
            to_print += 'Ave Standard PSO Selection Accuracy micro: %.4f\n' % np.average(
                sel_std_f1_mics)
            to_print += 'Ave Super PSO Selection Accuracy micro: %.4f\n' % np.average(
                sel_sup_f1_mics)
            to_print += 'Ave Super simple PSO Selection Accuracy  micro: %.4f\n' % np.average(
                sel_sim_f1_mics)

            to_print += 'Ave Full Accuracy macro: %.4f\n' % np.average(
                full_f1_macs)
            to_print += 'Ave Standard PSO Selection Accuracy macro: %.4f\n' % np.average(
                sel_std_f1_macs)
            to_print += 'Ave Super PSO Selection Accuracy macro: %.4f\n' % np.average(
                sel_sup_f1_macs)
            to_print += 'Ave Super simple PSO Selection Accuracy macro: %.4f\n' % np.average(
                sel_sim_f1_macs)

            to_print += 'Ave time of Standard PSO: %.4f\n' % np.average(
                PSO_std_durations)
            to_print += 'Ave time of Super PSO: %.4f\n' % np.average(
                PSO_sup_durations)
            to_print += 'Ave time of Super simple PSO: %.4f\n' % np.average(
                PSO_sim_durations)
            to_print += 'Ave Standard PSO Feature Ratio: %.2f\n' % np.average(
                f_std_ratios)
            to_print += 'Ave Super PSO Feature Ratio: %.2f\n' % np.average(
                f_sup_ratios)
            to_print += 'Ave Super simple PSO Feature Ratio: %.2f\n' % np.average(
                f_sim_ratios)

            f = open('records/fullstd_PSOsel_std_sup_simple_f1/r_' + datasets[i] + str(run) +
                     '_fullstd_PSOsel_std_sup_simple_f1.txt', 'w')
            f.write(to_print)
            f.close()


def fullstd_PSOsel_std_simple_f1(datasets_list, no_clses_list, run):
    from sklearn.metrics import f1_score
    for datasets, no_cls in zip(datasets_list, no_clses_list):
        for i in range(len(datasets)):
            X, y, feature_names, label_names = load_dataset(
                datasets[i], 'undivided')
            n_features = len(feature_names)
            X = pd.DataFrame.sparse.from_spmatrix(X).to_numpy()
            y = pd.DataFrame.sparse.from_spmatrix(y).to_numpy()

            n_splits = 5
            k_fold = IterativeStratification(
                n_splits=n_splits, order=1)

            full_f1_mics = []
            full_f1_macs = []

            sel_std_f1_mics = []
            sel_std_f1_macs = []
            sel_sim_f1_mics = []
            sel_sim_f1_macs = []

            PSO_std_durations = []
            PSO_sim_durations = []

            f_std_ratios = []
            f_sim_ratios = []
            to_print = ''
            fold_count = 0

            for train_idx, test_idx in k_fold.split(X, y):
                print('Fold ', fold_count)
                fold_count += 1

                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # get full acc
                # get standard classficaition acc(f1)
                scaler_1 = StandardScaler()
                scaler_2 = MinMaxScaler()
                X_train_scaled_1 = scaler_1.fit_transform(X_train)
                X_test_scaled_1 = scaler_1.fit_transform(X_test)
                X_train_scaled_2 = scaler_2.fit_transform(X_train_scaled_1)
                X_test_scaled_2 = scaler_2.fit_transform(X_test_scaled_1)

                clf = MLkNN(k=3)
                clf.fit(X_train_scaled_2, y_train)
                y_test_pred = clf.predict(X_test_scaled_2)

                full_f1_mic = f1_score(y_test, y_test_pred, average='micro')
                print('Full_f1_micro: ', full_f1_mic)
                full_f1_mics.append(full_f1_mic)

                full_f1_mac = f1_score(y_test, y_test_pred, average='macro')
                print('Full_f1_macro: ', full_f1_mac)
                full_f1_macs.append(full_f1_mac)

                #  perform standard PSO FS
                start_std_PSO = time.time_ns()    # marking start time of PSO
                problem_std = Problem.FS_ML_f1(
                    minimize=False, X=X_train, y=y_train)

                pop_size = 100
                n_iterations = 100
                swarm = Swarm.Swarm(n_particles=pop_size, length=n_features, pos_max=1.0, pos_min=0,
                                    vel_max=0.2, vel_min=-0.2, problem=problem_std, n_iterations=n_iterations)
                best_sol_std, best_fit_std = swarm.iterate()
                end_std_PSO = time.time_ns()      # marking ending time of PSO
                duration_std_PSO = round(
                    (end_std_PSO - start_std_PSO)/1000000000, 2)
                PSO_std_durations.append(duration_std_PSO)

                sel_fea_std = np.where(best_sol_std > problem_std.threshold)[0]
                clf.fit(X_train_scaled_2[:, sel_fea_std], y_train)
                y_test_pred = clf.predict(X_test_scaled_2[:, sel_fea_std])
                sel_std_f1_mic = f1_score(y_test, y_test_pred, average='micro')
                print('Selected_standard_f1_micro: ', sel_std_f1_mic)
                sel_std_f1_mics.append(sel_std_f1_mic)
                sel_std_f1_mac = f1_score(y_test, y_test_pred, average='macro')
                print('Selected_standard_f1_macro: ', sel_std_f1_mac)
                sel_std_f1_macs.append(sel_std_f1_mac)
                f_std_ratios.append(len(sel_fea_std)/n_features)


                # perform super simple PSO FS
                start_sim_PSO = time.time_ns()    # marking start time of PSO

                y_train_s = Super_simple.label_convert_simple(
                    y_train, no_cls)   # Convert y_train
                problem_sim = Problem.FS_ML_super_simple_f1(
                    minimize=False, X=X_train, y=y_train_s, no_cls=no_cls)

                pop_size = 100
                n_iterations = 100
                swarm = Swarm.Swarm(n_particles=pop_size, length=n_features, pos_max=1.0, pos_min=0,
                                    vel_max=0.2, vel_min=-0.2, problem=problem_sim, n_iterations=n_iterations)
                best_sol_sim, best_fit_sim = swarm.iterate()
                end_sim_PSO = time.time_ns()      # marking ending time of PSO
                duration_sim_PSO = round(
                    (end_sim_PSO - start_sim_PSO)/1000000000, 2)
                PSO_sim_durations.append(duration_sim_PSO)

                sel_fea_sim = np.where(best_sol_sim > problem_sim.threshold)[0]
                clf.fit(X_train_scaled_2[:, sel_fea_sim], y_train)
                y_test_pred = clf.predict(X_test_scaled_2[:, sel_fea_sim])
                sel_sim_f1_mic = f1_score(y_test, y_test_pred, average='micro')
                print('Selected_super_simple_f1_micro: ', sel_sim_f1_mic)
                sel_sim_f1_mics.append(sel_sim_f1_mic)
                sel_sim_f1_mac = f1_score(y_test, y_test_pred, average='macro')
                print('Selected_super_simple_f1_macro: ', sel_sim_f1_mac)
                sel_sim_f1_macs.append(sel_sim_f1_mac)
                f_sim_ratios.append(len(sel_fea_sim)/n_features)

                # to write the results
                to_print += '--------------Fold %d----------------\n' % fold_count
                to_print += 'Full feature f1 score micro: %.4f\n' % full_f1_mic
                to_print += 'Fold selected standard PSO f1 score micro: %.4f\n' % sel_std_f1_mic
                to_print += 'Fold selected super simple PSO f1 score micro: %.4f\n' % sel_sim_f1_mic

                to_print += 'Full feature f1 score macro: %.4f\n' % full_f1_mac
                to_print += 'Fold selected standard PSO f1 score macro: %.4f\n' % sel_std_f1_mac
                to_print += 'Fold selected super simple PSO f1 score macro: %.4f\n' % sel_sim_f1_mac

                to_print += 'Time of standard PSO: %.4f\n' % duration_std_PSO
                to_print += 'Time of super simple PSO: %.4f\n' % duration_sim_PSO
                to_print += 'Selection standard PSO ratio: %.2f\n' % (
                    len(sel_fea_std)/n_features)
                to_print += 'Selection super simple PSO ratio: %.2f\n' % (
                    len(sel_fea_sim)/n_features)
                to_print += 'Selected features standard PSO: %s\n' % (
                    ', '.join([str(ele) for ele in sel_fea_std]))
                to_print += 'Selected features super simple PSO: %s\n' % (
                    ', '.join([str(ele) for ele in sel_fea_sim]))

            to_print += '--------------Average----------------\n'
            to_print += 'Ave Full Accuracy micro: %.4f\n' % np.average(
                full_f1_mics)
            to_print += 'Ave Standard PSO Selection Accuracy micro: %.4f\n' % np.average(
                sel_std_f1_mics)
            to_print += 'Ave Super simple PSO Selection Accuracy  micro: %.4f\n' % np.average(
                sel_sim_f1_mics)

            to_print += 'Ave Full Accuracy macro: %.4f\n' % np.average(
                full_f1_macs)
            to_print += 'Ave Standard PSO Selection Accuracy macro: %.4f\n' % np.average(
                sel_std_f1_macs)
            to_print += 'Ave Super simple PSO Selection Accuracy macro: %.4f\n' % np.average(
                sel_sim_f1_macs)

            to_print += 'Ave time of Standard PSO: %.4f\n' % np.average(
                PSO_std_durations)
            to_print += 'Ave time of Super simple PSO: %.4f\n' % np.average(
                PSO_sim_durations)
            to_print += 'Ave Standard PSO Feature Ratio: %.2f\n' % np.average(
                f_std_ratios)
            to_print += 'Ave Super simple PSO Feature Ratio: %.2f\n' % np.average(
                f_sim_ratios)

            # f = open('records/full_PSOsel_std_sim_f1/record_' + str(run) + datasets[i] + 
            #          '_fullstd_PSOsel_std_simple_f1.txt', 'w')
            f = open(str(run) + datasets[i] + '_fullstd_PSOsel_std_sim_f1.txt', 'w')
            f.write(to_print)
            f.close()


def fullstd_PSOsel_std_simple_hl(datasets_list, no_clses_list, run):
    for datasets, no_cls in zip(datasets_list, no_clses_list):
        for i in range(len(datasets)):
            X, y, feature_names, label_names = load_dataset(
                datasets[i], 'undivided')
            n_features = len(feature_names)
            X = pd.DataFrame.sparse.from_spmatrix(X).to_numpy()
            y = pd.DataFrame.sparse.from_spmatrix(y).to_numpy()

            n_splits = 3
            k_fold = IterativeStratification(
                n_splits=n_splits, order=1)

            full_hams = []
            sel_std_hams = []
            sel_sim_hams = []

            PSO_std_durations = []
            PSO_sim_durations = []

            f_std_ratios = []
            f_sim_ratios = []
            to_print = ''
            fold_count = 0

            for train_idx, test_idx in k_fold.split(X, y):
                print('Fold ', fold_count)
                fold_count += 1

                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # get full acc
                # get standard classficaition acc(f1)
                scaler_1 = StandardScaler()
                scaler_2 = MinMaxScaler()
                X_train_scaled_1 = scaler_1.fit_transform(X_train)
                X_test_scaled_1 = scaler_1.fit_transform(X_test)
                X_train_scaled_2 = scaler_2.fit_transform(X_train_scaled_1)
                X_test_scaled_2 = scaler_2.fit_transform(X_test_scaled_1)

                clf = MLkNN(k=3)
                clf.fit(X_train_scaled_2, y_train)
                y_test_pred = clf.predict(X_test_scaled_2)

                full_ham = hamming_loss(y_test, y_test_pred)
                print('Full_ham: ', full_ham)
                full_hams.append(full_ham)

                #  perform standard PSO FS
                start_std_PSO = time.time_ns()    # marking start time of PSO
                problem_std = Problem.FS_ML_hl(
                    minimize=True, X=X_train, y=y_train)

                pop_size = 3
                n_iterations = 5
                swarm = Swarm.Swarm(n_particles=pop_size, length=n_features, pos_max=1.0, pos_min=0,
                                    vel_max=0.2, vel_min=-0.2, problem=problem_std, n_iterations=n_iterations)
                best_sol_std, best_fit_std = swarm.iterate()
                end_std_PSO = time.time_ns()      # marking ending time of PSO
                duration_std_PSO = round(
                    (end_std_PSO - start_std_PSO)/1000000000, 2)
                PSO_std_durations.append(duration_std_PSO)

                sel_fea_std = np.where(best_sol_std > problem_std.threshold)[0]
                clf.fit(X_train_scaled_2[:, sel_fea_std], y_train)
                y_test_pred = clf.predict(X_test_scaled_2[:, sel_fea_std])
                sel_std_ham = hamming_loss(y_test, y_test_pred)
                print('Selected_standard_ham: ', sel_std_ham)
                sel_std_hams.append(sel_std_ham)
                f_std_ratios.append(len(sel_fea_std)/n_features)


                # perform super simple PSO FS
                start_sim_PSO = time.time_ns()    # marking start time of PSO

                y_train_s = Super_simple.label_convert_simple(
                    y_train, no_cls)   # Convert y_train
                problem_sim = Problem.FS_ML_super_simple_hl(
                    minimize=True, X=X_train, y=y_train_s, no_cls=no_cls)

                pop_size = 3
                n_iterations = 5
                swarm = Swarm.Swarm(n_particles=pop_size, length=n_features, pos_max=1.0, pos_min=0,
                                    vel_max=0.2, vel_min=-0.2, problem=problem_sim, n_iterations=n_iterations)
                best_sol_sim, best_fit_sim = swarm.iterate()
                end_sim_PSO = time.time_ns()      # marking ending time of PSO
                duration_sim_PSO = round(
                    (end_sim_PSO - start_sim_PSO)/1000000000, 2)
                PSO_sim_durations.append(duration_sim_PSO)

                sel_fea_sim = np.where(best_sol_sim > problem_sim.threshold)[0]
                clf.fit(X_train_scaled_2[:, sel_fea_sim], y_train)
                y_test_pred = clf.predict(X_test_scaled_2[:, sel_fea_sim])
                sel_sim_ham = hamming_loss(y_test, y_test_pred)
                print('Selected_super_simple_ham: ', sel_sim_ham)
                sel_sim_hams.append(sel_sim_ham)
                f_sim_ratios.append(len(sel_fea_sim)/n_features)

                # to write the results
                to_print += '--------------Fold %d----------------\n' % fold_count
                to_print += 'Full feature f1 score micro: %.4f\n' % full_ham
                to_print += 'Fold selected standard PSO f1 score micro: %.4f\n' % sel_std_ham
                to_print += 'Fold selected super simple PSO f1 score micro: %.4f\n' % sel_sim_ham

                to_print += 'Time of standard PSO: %.4f\n' % duration_std_PSO
                to_print += 'Time of super simple PSO: %.4f\n' % duration_sim_PSO
                to_print += 'Selection standard PSO ratio: %.2f\n' % (
                    len(sel_fea_std)/n_features)
                to_print += 'Selection super simple PSO ratio: %.2f\n' % (
                    len(sel_fea_sim)/n_features)
                to_print += 'Selected features standard PSO: %s\n' % (
                    ', '.join([str(ele) for ele in sel_fea_std]))
                to_print += 'Selected features super simple PSO: %s\n' % (
                    ', '.join([str(ele) for ele in sel_fea_sim]))

            to_print += '--------------Average----------------\n'
            to_print += 'Ave Full Accuracy micro: %.4f\n' % np.average(
                full_hams)
            to_print += 'Ave Standard PSO Selection Accuracy micro: %.4f\n' % np.average(
                sel_std_hams)
            to_print += 'Ave Super simple PSO Selection Accuracy  micro: %.4f\n' % np.average(
                sel_sim_hams)

            to_print += 'Ave time of Standard PSO: %.4f\n' % np.average(
                PSO_std_durations)
            to_print += 'Ave time of Super simple PSO: %.4f\n' % np.average(
                PSO_sim_durations)
            to_print += 'Ave Standard PSO Feature Ratio: %.2f\n' % np.average(
                f_std_ratios)
            to_print += 'Ave Super simple PSO Feature Ratio: %.2f\n' % np.average(
                f_sim_ratios)

            f = open('records/full_PSOsel_std_sim_hl/record_' + datasets[i] + str(run) +
                     '_fullstd_PSOsel_std_simple_hl.txt', 'w')
            f.write(to_print)
            f.close()