import sys
sys.path.append('..')

import numpy as np
import pylab as pl
from loadData import *
from plotBoundary import *

def lin_pgs(X, Y, lmbd, max_epochs):
    t, epoch = 0, 0
    w_0 = 0
    w = np.zeros((X.shape[1], 1))

    while epoch < max_epochs:
        for i in range(len(X)):
            t += 1
            eta = 1. / (t * lmbd)
            if Y[i] * (X[i] @ w + w_0) < 1:
            # less than 1 from hinge loss
                w = (1 - eta * lmbd) * w + eta * Y[i] * X[i].reshape((-1, 1))
                w_0 = w_0 + eta * Y[i]
            else:
                w = (1 - eta * lmbd) * w
                w_0 = w_0
        epoch += 1
    return w_0, w

def rbf_pgs(X, Y, lmbd, gamma, max_epochs):
    t, epoch = 0, 0
    alpha = np.zeros((X.shape[0], 1))
    K = compute_rbf(X, gamma)

    while epoch < max_epochs:
        for i in range(len(X)):
            t += 1
            eta = 1. / (t * lmbd)
            disc = sum([alpha[j] * K[j][i] for j in range(len(alpha))])
            if Y[i] * disc < 1:
                alpha[i] = (1 - eta * lmbd) * alpha[i] + eta * Y[i]
            else:
                alpha[i] = (1 - eta * lmbd) * alpha[i]
        epoch += 1
    return alpha

def compute_rbf(X, gamma):
    K = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            K[i][j] = np.exp(-gamma * (np.linalg.norm(X[i] - X[j])) ** 2)
    return K

def margin(X, Y, w_0, w):
    return float(1. / np.sqrt(np.linalg.norm(w) ** 2 + w_0 ** 2))

def score(x, w_0, W):
    return float(x @ w + w_0)

def acc(X, Y, w_0, w):
    correct = 0
    for i in range(len(X)):
        if X[i] @ w + w_0 > 0:
            correct += Y[i] == 1
        else:
            correct += Y[i] == -1
    return float(correct / len(X))

def rbf_score(x, X_train, alpha, gamma):
    s = 0
    for i in range(len(X_train)):
        s += alpha[i] * exp(-gamma * np.linalg.norm(X_train[i] - x.reshape(-1,)) ** 2)
    return s

def rbf_acc(X, Y, X_train, alpha, gamma):
    correct = 0
    for i in range(len(X)):
        if rbf_score(X[i], X_train, alpha, gamma) > 0:
            correct += Y[i] == 1
        else:
            correct += Y[i] == -1
    return float(correct / len(X))


def part_one(plot=False):
    index = 3
    X, Y = load_train(index)
    X_val, Y_val = load_val(index)
    w_0, w = lin_pgs(X, Y, 0.01, 100)
    print('des: pegasos data3_train (λ = 0.01)')
    print('params: w_0, w w_0, w = {:.3f}, {:.3f}, {:.3f}'.format(w_0[0], w[0][0], w[1][0]))
    print('train_acc: {:.1%}'.format(acc(X, Y, w_0, w)))
    print('val_acc: {:.1%}'.format(acc(X_val, Y_val, w_0, w)))
    print()
    if plot:
        score = lambda x: x @ w + w_0
        plotDecisionBoundary(X, Y, score, [-1,0,1], title ='Gaussian Kernel SVM (λ = 0.01)')

def part_two():
    index = 3
    X, Y = load_train(index)
    for lmbd in [2**(-i) for i in range(-1,11)]:
        w_0, w = lin_pgs(X, Y, lmbd, 100)
        print('lambda = {:.1e}'.format(lmbd))
        print('margin = {:.3f}'.format(margin(X, Y, w_0, w)))
        print()

def part_three():
    pass

def part_four(plot=False):
    index = 3
    X, Y = load_train(index)
    X_val, Y_val = load_val(index)
    lmbd = 0.02
    max_epochs = 10

    gamma_trials = [4, 2, 1, 0.5, 0.25]

    for gamma in gamma_trials:
        alpha = rbf_pgs(X, Y, lmbd, gamma, max_epochs)
        supp = len(list(filter(lambda x: x != 0, alpha)))
        print('gamma = {}'.format(gamma))
        print('support vectors: {}'.format(supp))
        print('train_acc: {:.1%}'.format(rbf_acc(X, Y, X, alpha, gamma)))
        print('val_acc: {:.1%}'.format(rbf_acc(X_val, Y_val, X, alpha, gamma)))
        print()

        if plot:
            score = lambda x: rbf_score(x, X, alpha, gamma)
            plotDecisionBoundary(X, Y, score, [-1,0,1], title='Gaussian Kernel SVM (γ = {})'.format(gamma))

def main():
   # part_one(True)
   # part_two()
   # part_three()
   part_four(False)
   pl.show()

if __name__ == '__main__':
    main()
