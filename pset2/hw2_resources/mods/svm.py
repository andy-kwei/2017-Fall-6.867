import sys
sys.path.append('..')

import numpy as np
import pylab as pl
from sklearn.svm import SVC
from loadData import *
from plotBoundary import *

cmaps = [pl.cm.cool, pl.cm.bwr, pl.cm.coolwarm, pl.cm.winter]
colors = ['navy', 'blue', 'black', 'orange', 'turquoise', 'grey']

def sktrain(X, Y, C=1.0, kernel='linear', gamma='auto'):
    svm = SVC(C=C, kernel=kernel, gamma=gamma)
    svm.fit(X, Y.flatten())
    return svm

def acc(X, Y, svm):
    correct = 0
    for i in range(len(X)):
        if svm.predict([X[i]]) == Y[i]:
            correct += 1
    return correct / len(X)

def primal_coefs(X, Y, svm):
    w = np.zeros((X.shape[1], 1))
    for i, index in enumerate(svm.support_):
        w = w + svm.dual_coef_[0][i] * Y[index] * X[index].reshape(-1, 1)
    return w

def margin(X, Y, svm):
    w = primal_coefs(X, Y, svm)
    return 1. / np.linalg.norm(w)

def plot(X, Y, svm, title, cmap=None, color=None, show=False):
    params = primal_coefs(X, Y, svm)
    predictSVM = lambda x: x @ params
    plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title=title, colors=color, cmap=cmap)
    if show:
        pl.show()

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
    part_three()
    # pl.show()

def part_one(plot=False):
    X = np.array([[2, 2], [2, 3], [0, -1], [-3, -2]])
    Y = np.array([[1, 1, -1, -1]])
    svm = sktrain(X, Y, C=1)
    support = list(svm.support_vectors_)
    params = list(svm.coef_)
    print('Part 1: C-SVM (C = {})'.format(float(C)))
    print('Params: {}'.format(params))
    print('Support vectors: {}'.format(support))
    if plot:
        plot(X, Y, svm, 'Simple Case', show=show)

def part_two(plot=False):
    for i in [1, 2, 3, 4]:
        tag = 'data'+str(i)
        X_train, Y_train = load_train(i)
        X_val, Y_val = load_val(i)
        svm = sktrain(X_train, Y_train, C=1.0)
        params = svm.coef_
        train_acc = acc(X_train, Y_train, svm)
        val_acc = acc(X_val, Y_val, svm)
        summary(tag, params, train_acc, val_acc)
        if plot:
            plot(X_train, Y_train, svm, tag+'_train (C = 1)', color='orange', cmap=cmaps[1])
            plot(X_val, Y_val, svm, tag+'_validate (C = 1)', color='orange', cmap=cmaps[1])

def part_three(plot=False):
    index = 2
    X, Y = load_train(index)
    X_val, Y_val = load_val(index)

    for kernel, gamma in [('linear', 'auto'), ('rbf', 0.01), ('rbf', 0.1), 
                          ('rbf', 1.), ('rbf', 10), ('rbf', 100)]:
        for C in [0.01, 0.1, 1, 10, 100]:
            svm = sktrain(X, Y, C=C, kernel=kernel, gamma=gamma)

            if kernel == 'linear':
                tag = '{} kernel, C = {}'.format(kernel, C)
            else:
                tag = '{} kernel, gamma = {}, C = {}'.format(kernel, gamma, C)

            print('des: {}'.format(tag))
            print('margin: {:.3f}'.format(margin(X, Y, svm)))
            print('support vectors: {}'.format(sum(svm.n_support_)))
            print('train_acc: {:.1%}'.format(acc(X, Y, svm)))
            print('val_acc: {:.1%}'.format(acc(X_val, Y_val, svm)))
            print()
            if plot:
                plot(X, Y, svm, tag+' (data{}_train)'.format(index), color=colors[1], cmap=cmaps[2])

if __name__ == '__main__':
    main()