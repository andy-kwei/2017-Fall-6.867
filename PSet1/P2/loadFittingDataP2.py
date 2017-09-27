import sys
sys.path.append('..')

import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
import pandas as pd

import P1.loadFittingDataP1 as P1 

def getData(ifPlotData=True):
    # load the fitting data and (optionally) plot out for examination
    # return the X and Y as a tuple

    data = pl.loadtxt('curvefittingp2.txt')

    X = data[0,:]
    Y = data[1,:]

    if ifPlotData:
        plt.plot(X,Y,'o')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    return (X,Y)

def polynomial_basis(X, m):
    # X is a 1D array
    matrix = np.empty((len(X), m+1))
    for i in range(len(X)):
        for j in range(m+1):
            matrix[i][j] = X[i]**j
    return matrix

def cosine_basis(X, m):
    # X is a 1D array
    matrix = np.empty((len(X), m))
    for i in range(len(X)):
        for j in range(m):
            matrix[i][j] = np.cos(X[i] * (j+1) * np.pi)
    return matrix

def eval_poly(X, coeffs):
    coeffs_1D = coeffs.flatten()
    X_1D = X.flatten()
    Y = np.empty(shape=(len(X),))

    for i in range(len(X)):
        Y[i] = sum([coeffs_1D[n] * (X_1D[i] ** n) for n in range(len(coeffs_1D))])
    return Y

def eval_cos(X, coeffs):
    coeffs_1D = coeffs.flatten()
    X_1D = X.flatten()
    Y = np.empty(shape=(len(X),))

    for i in range(len(X)):
        Y[i] = sum([coeffs_1D[j] * (np.cos(X_1D[i] * (j+1) * np.pi)) \
                            for j in range(len(coeffs_1D))])
    return Y

def eval_actual(x):
    return np.cos(np.pi * x) + np.cos(2 * np.pi * x)

def beta_closed_form(X, Y):
    # X, Y are 2D arrays
    return (np.linalg.inv(X.T @ X) @ X.T @ Y.reshape(-1, 1))

def sum_sq_err(X, Y, beta):
    return np.sum((np.reshape(Y, (-1, 1)) \
        - X @ np.reshape(beta, (-1, 1)))**2)

def SSE_gradient(X, Y, beta):
    # gradient = -2[X]^T[Y] + 2[X]^T[X][\beta]
    # returns 2D array
    return -2 * X.T @ Y + 2 * X.T @ X @ beta

def part_one(save=True):
    plt.figure(1, figsize=(12, 3))

    X, Y = getData(False)
    M = [0,1,3,10]
    solutions = {}

    for i, m in enumerate(M):
        X_poly = polynomial_basis(X, m)
        beta = beta_closed_form(X_poly, Y)
        solutions[m] = beta

        x_values = np.linspace(0, 1, 100)
        y_values = eval_poly(x_values, beta)
        actual_y_values = eval_actual(x_values)

        plt.subplot(1, len(M), i+1)
        plt.plot(X, Y, 'ro')
        plt.plot(x_values, y_values)
        plt.plot(x_values, actual_y_values)
        plt.axis([0,1,-3,3])
        plt.gca().set_aspect(0.17, adjustable='box')
        plt.title('Polynomial Basis (m = '+str(m)+')')

    plt.tight_layout()
    # plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    if save:
        plt.savefig('fig/part_1.png')

    return solutions

def part_two():
    # don't have numerical gradient function yet
    # test list is currently empty
    X, Y = getData(False)
    test = []
    for beta in test:
        grad = SSE_gradient(beta)
        num_grad = numerical_gradient(beta, sum_sq_err)
        assert np.sum((grad - num_grad)**2) < 1e-5
    return True

def part_three():
    # can't do nothing right now unless I sklearn sklearn
    pass

def part_four(save=True):
    plt.figure(4, figsize=(12,3))

    X, Y = getData(False)
    M = [1,2,4,8]
    solutions = {}

    for i, m in enumerate(M):
        x_cos = cosine_basis(X, m)
        beta = beta_closed_form(x_cos, Y)
        solutions[m] = beta

        x_values = np.linspace(0, 1, 100)
        y_values = eval_cos(x_values, beta)
        actual_y_values = eval_actual(x_values)

        plt.subplot(1, len(M), i+1)
        plt.plot(X, Y, 'ro')
        plt.plot(x_values, y_values)
        plt.plot(x_values, actual_y_values)
        plt.axis([0,1,-3,3])
        plt.gca().set_aspect(0.17, adjustable='box')
        plt.title('Cosine Basis (m = '+str(m)+')')

    plt.tight_layout()
    # plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    if save:
        plt.savefig('fig/part_4.png')

    return solutions

def main():
    print(part_one())
    print(part_two())
    print(part_three())
    print(part_four())
    plt.show()

if __name__ == '__main__':
    main()

# print(sum_sq_err(np.array([[1]])))
# print(beta_closed_form(np.array([[1,6],[2,14],[4,20]]), \
# np.array([[2],[4],[6]])))