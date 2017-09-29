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

def getTrue():
    return pl.loadtxt('lasso_true_w.txt')

def lassoTrainData():
    return getData('lasso_train.txt')

def lassoValData():
    return getData('lasso_validate.txt')

def lassoTestData():
    return getData('lasso_test.txt')

def eval_basis(x, beta):
    wgt = beta.reshape(-1,)
    y = 0
    for i in range(13):
        if i == 0:
            y += wgt[i] * x
        else:
            y += wgt[i] * np.sin(x * 0.4 * np.pi * i) 
    return y

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

    if alpha == 0:
        X = basis(X)
        return np.linalg.inv(X.T @ X) @ X.T @ Y.reshape(-1, 1)

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
        trial_alphas = [1e-10, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
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

    return alpha, beta

def main():
    alpha, beta, sse = tune_alpha()

    print(beta)

    lasso_beta = beta.toarray()
    true_beta = getTrue()
    ols_beta = train(1e-10)

    ridge_beta = np.array([[ 1.57425225],
 [ 1.71528547],
 [ 2.20520313],
 [ 1.54470894],
 [ 0.85751108],
 [ 0.89255566],
 [ 1.22461672],
 [ 1.00554157],
 [ 0.19134684],
 [-0.38885778],
 [-0.18741277],
 [ 0.25136004],
 [-0.04394494]])


    x_values = np.linspace(-1, 1, 100)
    true_y_values = eval_basis(x_values, true_beta)
    lasso_y_values = eval_basis(x_values, lasso_beta)
    ridge_y_values = eval_basis(x_values, ridge_beta)
    ols_y_values = eval_basis(x_values, ols_beta)

    plt.figure(1, figsize=(8, 5))
    # plt.subplot(1, len(M), i+1)
    # plt.plot(X, Y, 'ro', label='data')
    plt.plot(x_values, lasso_y_values, label='lasso')
    plt.plot(x_values, ridge_y_values, label='ridge')
    plt.plot(x_values, ols_y_values, label='λ=0')
    plt.plot(x_values, true_y_values, label='true')
    plt.axis([-1,1,-8,8])

    x, y = lassoTrainData()
    plt.plot(x, y, 'o', label='Train')
    x, y = lassoValData()
    plt.plot(x, y, 'o', label='Val')
    x, y = lassoTestData()
    plt.plot(x, y, 'o', label='Test')

    plt.gca().set_aspect(0.07, adjustable='box')
    # plt.title('Lasso vs. Ridge')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figs/lasso_ridge.png')
    plt.show()
    # plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)


if __name__ == '__main__':
    # main()
    lasso_results()
