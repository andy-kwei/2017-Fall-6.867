import sys
sys.path.append('..')

import numpy as np
import pylab as pl
from sklearn.svm import SVC
import cvxopt as cvx
from loadData import *
from plotBoundary import *

cmaps = [pl.cm.cool, pl.cm.bwr, pl.cm.coolwarm, pl.cm.winter]
colors = ['navy', 'blue', 'black', 'orange', 'turquoise', 'grey']

def qp_dual(X, Y, C=1.0, kernel='linear', gamma=None):
    q = cvx.matrix(np.ones((len(X), 1)))
    G = cvx.matrix(np.concatenate(np.ones((len(X), 1)), np.ones((len(X), 1)) * -1))
    h = cvx.matrix(np.concatenate(np.ones((len(X), 1)) * C, np.zeros((len(X), 1))))
    A = cvx.matrix(Y.reshape((1, -1)))
    b = cvx.matrix(np.zeros((len(X), 1)))

    D = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        if kernel == 'linear':
            D[i][i] = -1./2 * X[i] @ X[i].reshape(-1,1) 
        elif kernel == 'rbf':
            D[i][i] = -1./2 
    P = cvx.matrix(D)

    return P, q, G, h, A, b

def qp_train(X, Y, C=1.0, kernel='linear', gamma=None):
    args = qp_dual(X, Y, C, kernel, gamma)
    solution = cvx.solvers.qp(*args)
    alpha = np.array(solution['x'])
    return alpha

def svm_train(X, Y, C=1.0, kernel='linear', gamma='auto'):
    svm = SVC(C=C, kernel=kernel, gamma=gamma)
    svm.fit(X, Y.flatten())
    return svm

def acc(X, Y, svm):
    correct = 0
    for i in range(len(X)):
        if svm.predict([X[i]]) == Y[i]:
            correct += 1
    return correct / len(X)

def margin(svm):
    return 1. / np.linalg.norm(svm.coef_)

def lin_score(x, svm):
    return float(x.reshape(1, -1) @ svm.coef_)

def rbf_score(x, X_train, gamma, svm):
    s = 0
    for i, index in enumerate(svm.support_):
        s += svm.dual_coef_[0][i] * exp(-gamma * np.linalg.norm(X_train[index] - x.reshape(-1,)) ** 2)
    return s

def lin_plot(X, Y, svm, title='', color='orange', cmap=pl.cm.bwr):
    predict = lambda x: lin_score(x, svm)
    plotDecisionBoundary(X, Y, predict, [-1, 0, 1], title=title, colors=color, cmap=cmap)

def rbf_plot(X, Y, X_train, gamma, svm, title='', color='orange', cmap=pl.cm.bwr):
    predict = lambda x: rbf_score(x, X_train, gamma, svm)
    plotDecisionBoundary(X, Y, predict, [-1, 0, 1], title=title, colors=color, cmap=cmap)

def summary(tag, params, train_acc, val_acc):
    np.set_printoptions(precision=3)
    print('tag: {}'.format(tag))
    print('params: {}'.format(params))
    print('train_acc: {:.2%}'.format(train_acc))
    print('val_acc: {:.2%}'.format(val_acc))
    print('')

def main():
    # part_one()
    # part_two()
    # part_three_lin()
    part_three_rbf(True)
    pl.show()

def part_one(plot=False):
    X = np.array([[2, 2], [2, 3], [0, -1], [-3, -2]])
    Y = np.array([[1, 1, -1, -1]])
    svm = svm_train(X, Y, C=1)
    support = list(svm.support_vectors_)
    params = list(svm.coef_)
    print('Part 1: C-SVM (C = {})'.format(float(C)))
    print('Params: {}'.format(params))
    print('Support vectors: {}'.format(support))
    if plot:
        lin_plot(X, Y, svm, 'Simple Case', show=show)

def part_two(plot=False):
    for i in [1, 2, 3, 4]:
        tag = 'data'+str(i)
        X, Y = load_train(i)
        X_val, Y_val = load_val(i)
        svm = svm_train(X, Y, C=1.0)
        params = svm.coef_
        train_acc = acc(X, Y, svm)
        val_acc = acc(X_val, Y_val, svm)
        summary(tag, params, train_acc, val_acc)
        if plot:
            lin_plot(X, Y, svm, tag+'_train (C = 1)', color=colors[3], cmap=cmaps[1])
            lin_plot(X_val, Y_val, svm, tag+'_validate (C = 1)', color=colors[3], cmap=cmaps[1])

def part_three_lin(plot=False):
    index = 2
    X, Y = load_train(index)
    X_val, Y_val = load_val(index)

    for C in [0.01, 0.1, 1, 10, 100]:
        svm = svm_train(X, Y, C=C)
        tag = 'Linear SVM (C = {})'.format(C)
        print('des: {}'.format(tag))
        print('support vectors: {}'.format(sum(svm.n_support_)))
        print('margin: {:.3f}'.format(margin(svm)))
        print('train_acc: {:.1%}'.format(acc(X, Y, svm)))
        print('val_acc: {:.1%}'.format(acc(X_val, Y_val, svm)))
        print()
        if plot:
            lin_plot(X, Y, svm, title=tag, color=colors[1], cmap=cmaps[2])

def part_three_rbf(plot=False):
    index = 2
    X, Y = load_train(index)
    X_val, Y_val = load_val(index)

    for gamma in [0.1, 1, 10]:
        for C in [0.01, 0.1, 1, 10, 100]:
            svm = svm_train(X, Y, C=C, gamma=gamma)
            tag = 'Gaussian Kernel SVM (γ = {}, C = {})'.format(gamma, C)
            print('des: {}'.format(tag))
            print('support vectors: {}'.format(sum(svm.n_support_)))
            print('train_acc: {:.1%}'.format(acc(X, Y, svm)))
            print('val_acc: {:.1%}'.format(acc(X_val, Y_val, svm)))
            print()
            if gamma == 0.1 and C == 0.01:
                rbf_plot(X, Y, X, gamma, svm, title=tag, color=colors[1], cmap=cmaps[2])

if __name__ == '__main__':
    main()