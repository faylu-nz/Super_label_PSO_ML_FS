import Problem
import Swarm
import Util
# import Super
import numpy as np

from sklearn.metrics import hamming_loss
from sklearn.preprocessing import StandardScaler
from skmultilearn.adapt import MLkNN
from skmultilearn.model_selection import IterativeStratification
import time
import warnings
warnings.filterwarnings('ignore')

# Main entry

if __name__ == '__main__':

    data = Util.read_arff('datasets/emotions.arff')
    y = data.iloc[:, 72:]
    X = data.iloc[:, :72]
    n_features = len(list(X))
    X = X.to_numpy()
    y = y.to_numpy()

    n_splits = 5
    k_fold = IterativeStratification(n_splits=n_splits, order=1)

    full_hams = []
    # full_super_hams = []
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

        # # calculate super label classification accuracy with full features
        # X_train_df, X_test_df, y_train_df, y_test_df = Util.Convert_to_df(
        #     X_train, X_test, y_train, y_test)
        # y_test_pred_sorted, y_test_ = Super.calc_preds(
        #     X_train_df, X_test_df, y_train_df, y_test_df)
        # full_super_ham = hamming_loss(y_test_, y_test_pred_sorted)
        # full_super_hams.append(full_super_ham)

        #  perform FS
        start_PSO = time.time_ns()    # marking start time of PSO
        #problem = Problem.FS_ML(minimize=True, X=X_train, y=y_train)
        problem = Problem.FS_ML_super(minimize=True, X=X_train, y=y_train)

        # parameter for PSO
        pop_size = 30
        n_iterations = 50
        swarm = Swarm.Swarm(n_particles=pop_size, length=n_features, pos_max=1.0, pos_min=0,
                            vel_max=0.2, vel_min=-0.2, problem=problem, n_iterations=n_iterations)
        best_sol, best_fit = swarm.iterate()
        end_PSO = time.time_ns()      # marking ending time of PSO
        duration_PSO = round((end_PSO - start_PSO)/1000000000, 2)
        PSO_durations.append(duration_PSO)

        # get full acc standard
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

        clf = MLkNN(k=3)
        clf.fit(X_train, y_train)
        y_test_pred = clf.predict(X_test)
        full_ham = hamming_loss(y_test, y_test_pred)
        full_hams.append(full_ham)

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
        # to_print += 'Full feature super hamming loss: %.4f\n' % full_super_ham
        to_print += 'Fold selected hamming loss: %.4f\n' % fold_ham
        # to_print += 'Time of standard PSO: %.4f\n' % duration_PSO
        to_print += 'Time of super label PSO: %.4f\n' % duration_PSO
        to_print += 'Selection ratio: %.2f\n' % (len(sel_fea)/n_features)
        to_print += 'Selected features: %s\n' % (
            ', '.join([str(ele) for ele in sel_fea]))

    to_print += '--------------Average----------------\n'
    to_print += 'Ave Full Accuracy: %.4f\n' % np.average(full_hams)
    # to_print += 'Ave Super Full Accuracy: %.4f\n' % np.average(full_super_hams)
    to_print += 'Ave Selection Accuracy: %.4f\n' % np.average(sel_hams)
    # to_print += 'Ave time of standard PSO: %.4f\n' % np.average(PSO_durations)
    to_print += 'Ave time of super label PSO: %.4f\n' % np.average(
        PSO_durations)
    to_print += 'Ave Feature Ratio: %.2f\n' % np.average(f_ratios)

    f = open('records/record1_super.txt', 'w')
    f.write(to_print)
    f.close()
