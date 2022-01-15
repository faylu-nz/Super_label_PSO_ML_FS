import Problem
import Swarm
import Util
import Super
import numpy as np

from sklearn.metrics import hamming_loss
from sklearn.preprocessing import StandardScaler
from skmultilearn.adapt import MLkNN
from skmultilearn.model_selection import IterativeStratification
import time
import warnings
warnings.filterwarnings('ignore')

datasets = ['birds-train', 'CAL500', 'emotions',
            'enron', 'flags', 'medical', 'scene', 'yeast']
locations = [260, 68, 72, 1001, 19, 1449, 294, 103]
n_labels = [19, 174, 6, 53, 7, 45, 6, 14]

# Main entry
if __name__ == '__main__':

    # Main entry

    data = Util.read_arff('datasets/' + datasets[3] + '.arff')

    y = data.iloc[:, locations[3]:]
    X = data.iloc[:, :locations[3]]
    n_features = len(list(X))
    X = X.to_numpy()
    y = y.to_numpy()

    n_splits = 5
    k_fold = IterativeStratification(
        n_splits=n_splits, order=1)

    full_hams = []
    super_full_hams = []
    sel_hams_std = []
    sel_hams_sup = []
    PSO_std_durations = []
    PSO_sup_durations = []
    f_ratios_std = []
    f_ratios_sup = []
    to_print = ''
    fold_count = 0

    for train_idx, test_idx in k_fold.split(X, y):
        print('Fold ', fold_count)
        fold_count += 1

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # get full standard acc
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.fit_transform(X_test)

        clf = MLkNN(k=3)
        clf.fit(X_train_scaled, y_train)
        y_test_pred = clf.predict(X_test_scaled)
        full_ham = hamming_loss(y_test, y_test_pred)
        full_hams.append(full_ham)

        # get full super classification accuracy
        X_train_df, X_test_df, y_train_df, y_test_df = Util.Convert_to_df(
            X_train, X_test, y_train, y_test)
        y_test_pred_super, y_test_super = Super.calc_preds(
            X_train_df, y_train_df, X_test_df, y_test_df)
        super_full_ham = hamming_loss(y_test_super, y_test_pred_super)
        super_full_hams.append(super_full_ham)

        #  perform FS
        pop_size = 3
        n_iterations = 5

        # Standard PSO
        start_PSO_std = time.time_ns()    # marking start time of PSO
        problem_std = Problem.FS_ML(minimize=True, X=X_train, y=y_train)
        # parameter for PSO
        swarm_std = Swarm.Swarm(n_particles=pop_size, length=n_features, pos_max=1.0, pos_min=0,
                                vel_max=0.2, vel_min=-0.2, problem=problem_std, n_iterations=n_iterations)
        best_sol_std, best_fit_std = swarm_std.iterate()
        end_PSO_std = time.time_ns()      # marking ending time of PSO
        duration_PSO_std = round((end_PSO_std - start_PSO_std)/1000000000, 2)
        PSO_std_durations.append(duration_PSO_std)
        # process the final solution
        sel_fea_std = np.where(best_sol_std > problem_std.threshold)[0]
        clf.fit(X_train[:, sel_fea_std], y_train)
        y_test_pred = clf.predict(X_test[:, sel_fea_std])
        fold_ham_std = hamming_loss(y_true=y_test, y_pred=y_test_pred)
        sel_hams_std.append(fold_ham_std)
        f_ratios_std.append(len(sel_fea_std)/n_features)

        # Super_label PSO
        start_PSO_sup = time.time_ns()    # marking start time of PSO
        problem_sup = Problem.FS_ML_super(minimize=True, X=X_train, y=y_train)
        # parameter for PSO
        swarm_sup = Swarm.Swarm(n_particles=pop_size, length=n_features, pos_max=1.0, pos_min=0,
                                vel_max=0.2, vel_min=-0.2, problem=problem_sup, n_iterations=n_iterations)
        best_sol_sup, best_fit_sup = swarm_sup.iterate()
        end_PSO_sup = time.time_ns()      # marking ending time of PSO
        duration_PSO_sup = round((end_PSO_sup - start_PSO_sup)/1000000000, 2)
        PSO_sup_durations.append(duration_PSO_sup)
        # process the final solution
        sel_fea_sup = np.where(best_sol_sup > problem_sup.threshold)[0]
        clf.fit(X_train[:, sel_fea_sup], y_train)
        y_test_pred = clf.predict(X_test[:, sel_fea_sup])
        fold_ham_sup = hamming_loss(y_true=y_test, y_pred=y_test_pred)
        sel_hams_sup.append(fold_ham_sup)
        f_ratios_sup.append(len(sel_fea_sup)/n_features)

        # to write the results
        to_print += '--------------Fold %d----------------\n' % fold_count
        to_print += 'Full feature standard hamming loss: %.4f\n' % full_ham
        to_print += 'Full feature super label hamming loss: %.4f\n' % super_full_ham
        to_print += 'Standard fold selected hamming loss: %.4f\n' % fold_ham_std
        to_print += 'Super label fold selected hamming loss: %.4f\n' % fold_ham_sup
        to_print += 'Time of standard PSO: %.4f\n' % duration_PSO_std
        to_print += 'Time of super label PSO: %.4f\n' % duration_PSO_sup
        to_print += 'Selection ratio standard: %.2f\n' % (
            len(sel_fea_std)/n_features)
        to_print += 'Selection ratio super label: %.2f\n' % (
            len(sel_fea_sup)/n_features)
        to_print += 'Selected features standard: %s\n' % (
            ', '.join([str(ele) for ele in sel_fea_std]))
        to_print += 'Selected features super label: %s\n' % (
            ', '.join([str(ele) for ele in sel_fea_sup]))

    to_print += '--------------Average----------------\n'
    to_print += 'Ave Standard Full Accuracy: %.4f\n' % np.average(full_hams)
    to_print += 'Ave Super label Full Accuracy: %.4f\n' % np.average(
        super_full_hams)
    to_print += 'Ave Standard Selection Accuracy: %.4f\n' % np.average(
        sel_hams_std)
    to_print += 'Ave Selection Super label Accuracy: %.4f\n' % np.average(
        sel_hams_sup)
    to_print += 'Ave time of standard PSO: %.4f\n' % np.average(
        PSO_std_durations)
    to_print += 'Ave time of super label PSO: %.4f\n' % np.average(
        PSO_sup_durations)
    to_print += 'Ave Standard Feature Ratio: %.2f\n' % np.average(f_ratios_std)
    to_print += 'Ave Super label Feature Ratio: %.2f\n' % np.average(
        f_ratios_sup)

    f = open('records/record_' + datasets[3] + '.txt', 'w')
    f.write(to_print)
    f.close()
