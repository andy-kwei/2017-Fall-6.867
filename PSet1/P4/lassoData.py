import sys
sys.path.append('..')

import pdb
import random
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from scipy.sparse import csr_matrix

import P2.loadFittingDataP2 as P2

# X is an array of N data points (one dimensional for now), that is, NX1
# Y is a Nx1 column vector of data values

def getData(name):
    data = pl.loadtxt(name)
    # Returns column matrices
    X = data[0:1].T
    Y = data[1:2].T
    return X, Y

def lassoTrainData():
    return getData('lasso_train.txt')

def lassoValData():
    return getData('lasso_validate.txt')

def lassoTestData():
    return getData('lasso_test.txt')

def basis(X):
    X = X.flatten()
    matrix = np.empty((len(X), 13))
    for i in range(len(X)):
        for j in range(13):
            if j == 0:
                matrix[i][j] = X[i]
            else:
                matrix[i][j] = np.sin(X[i] * 0.4 * np.pi * j)
    return matrix

def train(alpha):
    # returns a 2D array
    X, Y = lassoTrainData()
    lasso = Lasso(alpha=alpha, max_iter=100000)
    lasso.fit(basis(X), Y)
    beta = lasso.coef_.flatten()
    return beta

def validate(trial_alphas):
    X, Y = lassoValData()
    X_basis = basis(X)
    solutions = {}

    for alpha in trial_alphas:
        beta = train(alpha)
        solutions[alpha] = (csr_matrix(beta), P2.sum_sq_err(X_basis, Y, beta))
    return solutions

def tune_alpha():
    # returns a triple (alpha, trained beta, validation SSE) 
    # that gives the smallest validation SSE
    trial_alphas = np.linspace(1e-4, 1e-2, 1000)
    solutions = validate(trial_alphas)
    best = (None, None, np.inf)

    for alpha in solutions:
        if solutions[alpha][1] < best[2]:
            best = (alpha, solutions[alpha][0], solutions[alpha][1])
    return best

def test(beta):
    X, Y = lassoTestData()
    return P2.sum_sq_err(basis(X), Y, beta)

def lasso_results(write_to_file=False):
    try:
        if write_to_file: 
            sys.stdout=open("results.txt", "w")

        print('Results:')
        trial_alphas = [1e-8, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
        betas = validate(trial_alphas)
        for alpha in trial_alphas:
            print('alpha =', alpha, ':')
            print(betas[alpha][0])
            print('Validation SSE =', betas[alpha][1], '\n')

        alpha, beta, val_sse = tune_alpha()
        print('Optimal params: \nalpha = {alpha} \nbeta = \n{beta} \nValidation ' \
            'SSE = {val_sse} \n'.format(alpha=alpha, beta=beta, val_sse=val_sse))
        
        print('Testing optimal params:')
        print('Test SSE =', test(beta.toarray()), '\n')

    finally:
        sys.stdout.close()

def main():
    lasso_results()

if __name__ == '__main__':
    main()
