import pylab as pl
import loadParametersP1 as lp1
import numpy as np
import matplotlib.pyplot as plt
from math import *
import random

def SLGrad(x, y, theta):
    return 2 * x * (x.T * theta - y)

def makeBatchGrad(xs, ys):
    def batchGrad(theta):
        out = np.zeros_like(theta)
        for i in range(len(xs)):
            out += SLGrad(xs[i].T, ys[i], theta)
        return out
    return batchGrad

index = -1
def makeStochGrad(xs, ys):
    shuffle = list(range(len(xs)))
    def stochGrad(theta):
        global index
        if not 0 <= index < len(xs):
            random.shuffle(shuffle)
            index = 0
        i = shuffle[index]
        index += 1
        return SLGrad(xs[i].T, ys[i], theta)
    return stochGrad

def makeValue(X, y):
    def value(theta):
        return np.linalg.norm(X * theta - y)
    return value

def getData():
    
    # load the fitting data for X and y and return as elements of a tuple
    # X is a 100 by 10 matrix and y is a vector of length 100
    # Each corresponding row for X and y represents a single data sample

    X = pl.loadtxt('fittingdatap1_x.txt')
    y = pl.loadtxt('fittingdatap1_y.txt')

    return (X,y) 

if __name__ == '__main__':
    X, y = getData()
    xs = np.mat(X)
    ys = y
    
    start = np.ones((10, 1))

    batchGrad = makeBatchGrad(xs, ys)
    stochGrad = makeStochGrad(xs, ys)

    value = makeValue(np.mat(X), np.mat(y))

    # print(value(start))
    
    ## PLOTS
    
    rateFun = lambda iterations: 1e-4 *(1+ iterations) ** -0.75
    steps = []
    gradFun = batchGrad
    convergence_specs = (value,1e-10,"objective")
    a = lp1.gradientDescent(gradFun, start, rateFun, convergence_specs, steps)
    plt.title("Batch Gradient Descent")
    plt.xlabel("Iterations")
    plt.ylabel("Difference from Goal Objective Function")
    # print(steps[0])
    # print(a)
    # print('err here')
    plt.plot([x for x, y,z in steps], [abs(z) for x, y,z in steps])
    plt.show()
    
    rateFun = lambda iterations: 1e-7 *(1+ iterations) ** -0.75
    steps = []
    gradFun = stochGrad
    convergence_specs = (value,5e-6,"objective")
    a = lp1.gradientDescent(gradFun, start, rateFun, convergence_specs, steps)
    plt.title("Stochastic Gradient Descent")
    plt.xlabel("Iterations")
    plt.ylabel("Difference from Goal Objective Function")
    # print(steps[0])
    # print(a)
    # print('err here')
    plt.plot([x for x, y,z in steps], [abs(z) for x, y,z in steps])
    #plt.plot([x for x, y,z in steps], [log(np.linalg.norm(gradFun(y))) for x, y,z in steps])
    plt.show()