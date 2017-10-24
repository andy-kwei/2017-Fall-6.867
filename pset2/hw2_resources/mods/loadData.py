import numpy as np

def load_train(index):
    data = np.loadtxt('../data/data'+str(index)+'_train.csv')
    X = data[:, 0:2].copy()
    Y = data[:, 2:3].copy()
    return X, Y

def load_val(index):
    data = np.loadtxt('../data/data'+str(index)+'_validate.csv')
    X = data[:, 0:2].copy()
    Y = data[:, 2:3].copy()
    return X, Y

def load_test(index):
    data = np.loadtxt('../data/data'+str(index)+'_test.csv')
    X = data[:, 0:2].copy()
    Y = data[:, 2:3].copy()
    return X, Y

def load_mnist(digit, norm=True):
    X = np.loadtxt('../data/mnist_digit_'+str(digit)+'.csv')
    if norm:
        X = 2. * X / 255 - np.ones_like(X)
    X_train = X[:200, :]
    X_val = X[200:350, :]
    X_test = X[350:500, :]
    return X_train, X_val, X_test