import numpy as np
from sklearn.linear_model import LogisticRegression
from . import gradientDescent as gd

def sktrain(X, Y, reg='l2', alpha=1.):
    lr = LogisticRegression(penalty=reg, C=1./alpha)
    lr.fit(X, Y)
    return np.concatenate((lr.intercept_, lr.coef_.reshape(-1,)))

def train(X, Y, reg='l2', alpha=1., eta=1., epsilon=1e-8, max_iters=1e4, gd_method='batch'):
    obj = logistic_loss_fn(X, Y, reg=reg, alpha=alpha)
    grad = gd.num_grad_fn(obj)
    grad_gen = lambda i: (lambda w: gd.num_grad_fn(
        logistic_loss_fn(X, Y, reg=reg, alpha=alpha, index=i))(w))
    start = np.ones((X.shape[1]+1, 1))

    if isinstance(eta, float):
        const = eta
        eta = lambda x: const/(1+x)**0.8

    if gd_method == 'batch':
        sol = gd.batch(grad, start, eta=eta, epsilon=epsilon, max_iters=max_iters)
    elif gd_method == 'sgd':
        sol = gd.sgd(grad_gen, start, eta=eta, epsilon=epsilon, max_iters=max_iters)
    else:
        print('Parameter gd_method incorrect.')
        sol = None
    return sol

def logistic_loss_fn(X, Y, reg='l2', alpha=1., index=None):
    def L(weights):
        loss = 0
        if reg == 'l2':
            loss += alpha * (weights.T @ weights).item()
        elif reg == 'l1':
            loss += alpha * sum(abs(weights))

        w_0 = weights[:1,:].item()
        w = weights[1:,:]

        if index is None:
            for i in range(X.shape[0]):
                z = Y[i].item() * ((X[i] @ w).item() + w_0)
                loss += np.log(1+np.exp(-z))
        else:
            i = index % X.shape[0]
            z = Y[i].item() * ((X[i] @ w).item() + w_0)
            loss += np.log(1+np.exp(-z))
        return loss
    return L

def logistic_grad(X, Y, w, index=None):
    # Only for unregularized LR
    X_padded = np.ones((X.shape[0], X.shape[1]+1))
    X_padded[:,1:] = X
    grad = np.zeros((X.shape[1]+1, 1))

    if index is None:
        for i in range(len(grad)):
            for j in range(X.shape[0]):
                z = Y[j] * (X_padded[j] @ w)
                z_prime = Y[j] * X_padded[j][i]
                grad[i][0] += -1. / (1 + np.exp(z)) * z_prime
    else:
        for i in range(len(grad)):
            j = index % X.shape[0]
            z = Y[j] * (X_padded[j] @ w)
            z_prime = Y[j] * X_padded[j][i]
            grad[i][0] = -1. / (1 + np.exp(z)) * z_prime
    return grad