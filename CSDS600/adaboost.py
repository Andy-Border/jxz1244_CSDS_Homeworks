import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from libsvm.svmutil import *
from libsvm.svm import *
import numpy as np
from sklearn.svm import LinearSVC


def get_error_rate(pred, Y):
    return sum(pred != Y) / len(pred)


class AdaBoost:
    def __init__(self, T=20):
        self.T = T

    def adaboost(self, X_train, Y_train, X_test, Y_test):
        # init
        Linear_SVM = LinearSVC()
        n_train, n_test = len(X_train), len(X_test)
        distribute_weight = np.ones(n_train) / n_train
        pred_train, pred_test, pred_test_score = np.zeros(n_train), np.zeros(n_test), np.zeros(n_test)
        T = self.T

        for i in range(T):
            # fit a base classifier
            Linear_SVM.fit(X_train, Y_train, sample_weight=distribute_weight)
            temp_pred_train = Linear_SVM.predict(X_train)
            temp_pred_test = Linear_SVM.predict(X_test)
            # temp_pred_test_score = dec_tree.predict_proba(X_test)

            miss = [int(x) for x in temp_pred_train != Y_train]
            loss = np.dot(distribute_weight, miss)
            if loss > 0.5:
                break
            alpha = 0.5 * np.log(1 / loss - 1)
            # add to prediction
            pred_train += alpha * temp_pred_train
            pred_test += alpha * temp_pred_test
            # pred_test_score += alpha * temp_pred_test_score[:, 1].ravel()
            # update distribution_weight
            params = [1 if x == 1 else -1 for x in miss]
            distribute_weight = distribute_weight * [np.exp(alpha * x) for x in params]
            distribute_weight = distribute_weight * (1 / sum(distribute_weight))

        pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)

        # print(get_error_rate(pred_test, Y_test))
        return pred_test


if __name__ == '__main__':

    datapath = 'DogsVsCats/DogsVsCats/'
    train_y, train_x = svm_read_problem(f'{datapath}DogsVsCats.train')
    test_y, test_x = svm_read_problem(f'{datapath}DogsVsCats.test')

    X_train = []
    for x in train_x:
        x = list(x.values())
        X_train.append(x)
    X_train = np.array(X_train)

    X_test = []
    for x in test_x:
        x = list(x.values())
        X_test.append(x)
    X_test = np.array(X_test)

    Y_train = np.array(train_y).astype(float)
    Y_test = np.array(test_y).astype(float)


    Linear_SVM = LinearSVC()
    Linear_SVM.fit(X_train, Y_train)
    pred_test = Linear_SVM.predict(X_test)
    print('accuracy on test data (standard decision tree): %f, error rata: %f' % (
        accuracy_score(Y_test, pred_test), get_error_rate(pred_test, Y_test)))


    ada_boost = AdaBoost()
    pred_test = ada_boost.adaboost(X_train, Y_train, X_test, Y_test)
    print('T = %d, accuracy on test data: %f, error rate: %f' % (
        ada_boost.T, accuracy_score(Y_test, pred_test), get_error_rate(pred_test, Y_test)))