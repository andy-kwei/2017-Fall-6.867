import sys
sys.path.append('../P1')

import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
import pandas as pd
import sys
from random import randint

import loadParametersP1 as P1P
import loadFittingDataP1 as P1F

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
    coeffs = coeffs.flatten()
    X = X.flatten()
    Y = np.empty(shape=(len(X),))

    for i in range(len(X)):
        Y[i] = sum([coeffs[n] * (X[i] ** n) for n in range(len(coeffs))])
    return Y

def eval_cos(X, coeffs):
    coeffs = coeffs.flatten()
    X = X.flatten()
    Y = np.empty(shape=(len(X),))

    for i in range(len(X)):
        Y[i] = sum([coeffs[j] * (np.cos(X[i] * (j+1) * np.pi)) \
                            for j in range(len(coeffs))])
    return Y

def eval_actual(x):
    return np.cos(np.pi * x) + np.cos(2 * np.pi * x)

def beta_closed_form(X, Y):
    # X, Y are 2D arrays
    return (np.linalg.inv(X.T @ X) @ X.T @ Y.reshape(-1, 1))

def sum_sq_err(X, Y, beta):
    return np.sum((np.reshape(Y, (-1, 1)) \
        - X @ np.reshape(beta, (-1, 1)))**2)

def sse_gradient(X, Y, beta):
    # X, Y are 2D arrays
    # gradient = -2[X]^T[Y] + 2[X]^T[X][\beta]
    # returns 2D array
    return -2 * X.T @ Y + 2 * X.T @ X @ beta

def get_obj_func(X, Y):
    def obj(beta):
        return sum_sq_err(X, Y, beta)
    return obj

def get_batch_grad_func(X, Y):
    def grad(beta):
        return sse_gradient(X, Y, beta)
    return grad

# global i
# i = 0
def get_sgd_grad_func(X, Y):
    def grad(beta):
        # global i
        # i = (i+1) % X.shape[1]
        i = randint(0, X.shape[1] - 1)
        return sse_gradient(X[i].reshape(1, -1), Y[i].reshape(1, -1), beta)
    return grad

def part_one(save=True, plot=True):
    X, Y = getData(False)
    M = [0,1,3,10]
    solutions = {}

    for i, m in enumerate(M):
        X_poly = polynomial_basis(X, m)
        beta = beta_closed_form(X_poly, Y)
        solutions[m] = beta

        if plot:
            x_values = np.linspace(0, 1, 100)
            y_values = eval_poly(x_values, beta)
            actual_y_values = eval_actual(x_values)
            plt.figure(1, figsize=(6,6))
            plt.subplot(2, 2, i+1)
            plt.plot(X, Y, 'ro', label='data')
            plt.plot(x_values, y_values, label='reg fit')
            plt.plot(x_values, actual_y_values, label='source')
            plt.axis([0,1,-3,3])
            plt.gca().set_aspect(0.17, adjustable='box')
            plt.title('Polynomial Fit (M = '+str(m)+')')
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

            if i == 0:
                plt.legend()

    if save:
        plt.savefig('fig/part_1.png')

    return solutions

def part_two():
    # test list can be expanded
    X, Y = getData(False)
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    test = [np.array([11]).reshape(1,1)]
    stepSize = 1

    for beta in test:
        closed_form_grad = sse_gradient(X, Y, beta)
        numerical_grad = P1P.centralDifferences(get_obj_func(X, Y), stepSize)(beta)
        assert np.sum((closed_form_grad - numerical_grad)**2) < 1e-5

    return closed_form_grad, np.array(numerical_grad)

def part_three_bgd(eta=0.01, threshold=1e-7, start=None, \
                    save=True, plot=True):
    # bgd 
    X, Y = getData(False)
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)

    M = [0,1,3,10]
    solutions = {}
    for i, m in enumerate(M):
        X_poly = polynomial_basis(X, m)
        obj = get_obj_func(X_poly, Y)
        bgd = get_batch_grad_func(X_poly, Y)

        if start == None:
            seed = np.zeros((X_poly.shape[1], 1))
        else:
            seed = start

        # calling gradient descent from P1
        beta = P1P.gradientDescent(bgd, seed, eta, \
            (obj, threshold, 'objective'), [])
        solutions[m] = beta

        if plot:
            x_values = np.linspace(0, 1, 100)
            y_values = eval_poly(x_values, beta)
            actual_y_values = eval_actual(x_values)

            plt.figure(2, figsize=(6,6))
            plt.subplot(2, 2, i+1)
            plt.plot(X, Y, 'ro', label='data')
            plt.plot(x_values, y_values, label='bgd fit')
            plt.plot(x_values, actual_y_values, label='source')
            plt.axis([0,1,-3,3])
            plt.gca().set_aspect(0.17, adjustable='box')
            plt.title('Batch GD (M = '+str(m)+')')
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

            if i == 0:
                plt.legend()

    if save:
        plt.savefig('fig/part_3_bgd.png')

    return solutions

def part_three_sgd(eta=0.01, threshold=1e-7, start=None, \
                    save=True, plot=True):
    # sgd 
    X, Y = getData(False)
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)

    M = [0,1,3,10]
    solutions = {}
    for i, m in enumerate(M):
        X_poly = polynomial_basis(X, m)
        obj = get_obj_func(X_poly, Y)
        sgd = get_sgd_grad_func(X_poly, Y)

        if start == None:
            seed = np.zeros((X_poly.shape[1], 1))
        else:
            seed = start

        # calling gradient descent from P1
        beta = P1P.gradientDescent(sgd, seed, eta, \
            (obj, threshold, 'objective'), [])
        solutions[m] = beta

        if plot:
            x_values = np.linspace(0, 1, 100)
            y_values = eval_poly(x_values, beta)
            actual_y_values = eval_actual(x_values)

            plt.figure(3, figsize=(6,6))
            plt.subplot(2, 2, i+1)
            plt.plot(X, Y, 'ro', label='data')
            plt.plot(x_values, y_values, label='sgd fit')
            plt.plot(x_values, actual_y_values, label='source')
            plt.axis([0,1,-3,3])
            plt.gca().set_aspect(0.17, adjustable='box')
            plt.title('Stochastic GD (M = '+str(m)+')')
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

            if i == 0:
                plt.legend()

    if save:
        plt.savefig('fig/part_3_sgd.png')

    return solutions

def part_three(save=True, plot=True):
    return part_three_bgd(save=save, plot=plot), \
        part_three_sgd(save=save, plot=plot)

def part_four(save=True, plot=True):
    X, Y = getData(False)
    M = [1,2,4,8]
    solutions = {}

    for i, m in enumerate(M):
        x_cos = cosine_basis(X, m)
        beta = beta_closed_form(x_cos, Y)
        solutions[m] = beta

        if plot:
            x_values = np.linspace(0, 1, 100)
            y_values = eval_cos(x_values, beta)
            actual_y_values = eval_actual(x_values)

            plt.figure(4, figsize=(6,6))
            plt.subplot(2, 2, i+1)
            plt.plot(X, Y, 'ro', label='data')
            plt.plot(x_values, y_values, label='reg fit')
            plt.plot(x_values, actual_y_values, label='source')
            plt.axis([0,1,-3,3])
            plt.gca().set_aspect(0.17, adjustable='box')
            plt.title('Cosine Basis (M = '+str(m)+')')
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

            if i == 0:
                plt.legend()

    if save:
        plt.savefig('fig/part_4.png')

    return solutions

def generate_results(show_plot=False, write_to_file=False):
    try:
        if write_to_file:
            sys.stdout = open('results.txt', 'w')

        part_1 = part_one(save=False, plot=False)
        part_2 = part_two()
        part_3 = part_three(save=False, plot=True)
        part_4 = part_four(save=False, plot=False)

        print('Part 1: Coefficients to polynomial basis regression:\n{}\n'.format(
            part_1))
        print('Part 2: Closed-form gradient vs. numerical gradient:\n{}\n'.format(
            part_2))
        print('Part 3: Batch GD vs. Stochastic GD\n{}\n'.format(part_3))
        print('Part 4: Coefficients to cosine basis regression:\n{}\n'.format(
            part_4))
    finally:
        sys.stdout.close()

    if show_plot:
        plt.show()

def main():
    part_three_bgd()
    part_three_sgd()
    plt.show()
    # generate_results(show_plot=True, write_to_file=False)

if __name__ == '__main__':
    main()
