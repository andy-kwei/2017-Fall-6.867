import sys
sys.path.append('..')

import numpy as np
import pylab as pl
from loadData import *
import svm as svm
import pegasos as pgs
import logisticRegression as lr
from plotBoundary import *

cache = {}

def main():
    args = [[1], [7], True]
    lr_classifer(*args)
    lin_svm_classifer(*args)
    rbf_svm_classifer(*args)
    lin_pgs_classifer(*args)
    rbf_pgs_classifer(*args)

def rbf_pgs_classifer(group_1, group_2, norm=True):
    X_tr, Y_tr, X_vl, Y_vl, X_ts, Y_ts = classify(group_1, group_2, norm)

    alpha, lmbd_star, gamma_star, hi = None, None, None, 0
    for lmbd in [0.02, 0.2, 2]:
        for gamma in [0.02, 0.2, 2]:
            tmp_alpha = pgs.rbf_pgs(X_tr, Y_tr, lmbd, gamma, max_epochs=10)
            val_acc = pgs.rbf_acc(X_vl, Y_vl, X_tr, tmp_alpha, gamma)
            if val_acc > hi:
                hi = val_acc
                lmbd_star = lmbd
                gamma_star = gamma
                alpha = tmp_alpha

    train_acc = pgs.rbf_acc(X_tr, Y_tr, X_tr, alpha, gamma)
    val_acc = pgs.rbf_acc(X_vl, Y_vl, X_tr, alpha, gamma)
    test_acc = pgs.rbf_acc(X_ts, Y_ts, X_tr, alpha, gamma)

    print('Pegasos Gaussian RBF SVM')
    print('λ*, γ* = {:.2e}, {:.2e}'.format(lmbd_star, gamma_star))
    print('train_acc: {:.1%}'.format(train_acc))
    print('val_acc: {:.1%}'.format(val_acc))
    print('test_acc: {:.1%}'.format(test_acc))
    print()
    return train_acc, val_acc, test_acc

def lin_pgs_classifer(group_1, group_2, norm=True):
    X_tr, Y_tr, X_vl, Y_vl, X_ts, Y_ts = classify(group_1, group_2, norm)

    w_0, w, lmbd_star, hi = None, None, None, 0
    for lmbd in [2**i for i in range(-12, 2, 1)]:
        tmp_w_0, tmp_w = pgs.lin_pgs(X_tr, Y_tr, lmbd, max_epochs=1000)       
        val_acc = pgs.acc(X_vl, Y_vl, tmp_w_0, tmp_w)
        if val_acc > hi:
            hi = val_acc
            lmbd_star = lmbd
            w_0 = tmp_w_0
            w = tmp_w
        
    train_acc = pgs.acc(X_tr, Y_tr, w_0, w)
    val_acc = pgs.acc(X_vl, Y_vl, w_0, w)
    test_acc = pgs.acc(X_ts, Y_ts, w_0, w)

    print('Pegasos Linear SVM')
    print('λ* = {:.2e}'.format(lmbd_star))
    print('train_acc: {:.1%}'.format(train_acc))
    print('val_acc: {:.1%}'.format(val_acc))
    print('test_acc: {:.1%}'.format(test_acc))
    print()
    return train_acc, val_acc, test_acc

def rbf_svm_classifer(group_1, group_2, norm=True):
    X_tr, Y_tr, X_vl, Y_vl, X_ts, Y_ts = classify(group_1, group_2, norm)

    s, C_star, gamma_star, hi = None, None, None, 0
    for C in [0.02, 0.2, 2]:
        for gamma in [0.02, 0.2, 2]:
            tmp_s = svm.svm_train(X_tr, Y_tr, kernel='rbf', C=C, gamma=gamma)
            val_acc = svm.acc(X_vl, Y_vl, tmp_s)
            if val_acc > hi:
                hi = val_acc
                C_star = C
                gamma_star = gamma
                s = tmp_s

    train_acc = svm.acc(X_tr, Y_tr, s)
    val_acc = svm.acc(X_vl, Y_vl, s)
    test_acc = svm.acc(X_ts, Y_ts, s)

    print('QP Gaussian RBF SVM')
    print('C*, γ* = {:.2e}, {:.2e}'.format(C_star, gamma_star))
    print('train_acc: {:.1%}'.format(train_acc))
    print('val_acc: {:.1%}'.format(val_acc))
    print('test_acc: {:.1%}'.format(test_acc))
    print()
    return train_acc, val_acc, test_acc


def lin_svm_classifer(group_1, group_2, norm=True):
    X_tr, Y_tr, X_vl, Y_vl, X_ts, Y_ts = classify(group_1, group_2, norm)
    
    s, C_star, hi = None, None, 0
    for C in [2**i for i in range(-12, 2, 1)]:
        tmp_s = svm.svm_train(X_tr, Y_tr, C=C)
        val_acc = svm.acc(X_vl, Y_vl, tmp_s)
        if val_acc > hi:
            hi = val_acc
            C_star = C
            s = tmp_s

    train_acc = svm.acc(X_tr, Y_tr, s)
    val_acc = svm.acc(X_vl, Y_vl, s)
    test_acc = svm.acc(X_ts, Y_ts, s)

    print('QP Linear SVM')
    print('C* = {:.2e}'.format(C_star))
    print('train_acc: {:.1%}'.format(train_acc))
    print('val_acc: {:.1%}'.format(val_acc))
    print('test_acc: {:.1%}'.format(test_acc))
    print()
    return train_acc, val_acc, test_acc

def lr_classifer(group_1, group_2, norm=True):
    X_tr, Y_tr, X_vl, Y_vl, X_ts, Y_ts = classify(group_1, group_2, norm)

    w, reg_star, lmbd_star, hi = None, None, None, 0
    for reg in ['l1', 'l2']:
        for lmbd in [2**i for i in range(-12, 2, 1)]:
            tmp_w = lr.lr_train(X_tr, Y_tr, reg, lmbd)
            val_acc = lr.acc(X_vl, Y_vl, tmp_w)
            if val_acc > hi:
                hi = val_acc
                reg_star = reg
                lmbd_star = lmbd
                w = tmp_w

    train_acc = lr.acc(X_tr, Y_tr, w)
    val_acc = lr.acc(X_vl, Y_vl, w)
    test_acc = lr.acc(X_ts, Y_ts, w)

    print('Logistic Regression')
    print('{} reg (λ* = {:.2e})'.format(reg_star.upper(), lmbd_star))
    print('train_acc: {:.1%}'.format(train_acc))
    print('val_acc: {:.1%}'.format(val_acc))
    print('test_acc: {:.1%}'.format(test_acc))
    print()
    return train_acc, val_acc, test_acc

def classify(group_1, group_2, norm=True):
    if (tuple(group_1), tuple(group_2)) not in cache:
        g1_train, g1_val, g1_test = np.empty((0, 784)), np.empty((0, 784)), np.empty((0, 784))
        g2_train, g2_val, g2_test = np.empty((0, 784)), np.empty((0, 784)), np.empty((0, 784))
        
        for d in group_1:
            X_train, X_val, X_test = load_mnist(d, norm)
            g1_train = np.concatenate((X_train, g1_train))
            g1_val = np.concatenate((g1_val, X_val))
            g1_test = np.concatenate((g1_test, X_test))

        for d in group_2:
            X_train, X_val, X_test = load_mnist(d, norm)
            g2_train = np.concatenate((g2_train, X_train))
            g2_val = np.concatenate((g2_val, X_val))
            g2_test = np.concatenate((g2_test, X_test))

        X_train = np.concatenate((g1_train, g2_train))
        Y_train = np.concatenate((np.ones((len(g1_train), 1)), -1. * np.ones((len(g2_train), 1))))

        X_val = np.concatenate((g1_val, g2_val))
        Y_val = np.concatenate((np.ones((len(g1_val), 1)), -1. * np.ones((len(g2_val), 1))))

        X_test = np.concatenate((g1_test, g2_test))
        Y_test = np.concatenate((np.ones((len(g1_test), 1)), -1. * np.ones((len(g2_test), 1))))

        cache[(tuple(group_1), tuple(group_2))] = X_train, Y_train, X_val, Y_val, X_test, Y_test
    return cache[(tuple(group_1), tuple(group_2))]

if __name__ == '__main__':
    main()