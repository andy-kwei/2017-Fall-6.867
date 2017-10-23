from numpy import *
from plotBoundary import *
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import mods.gradientDescent as gd
import mods.logisticRegression as lr

def main():
    plotPath()

def accuracy(X, Y, params):
    errors = 0
    for i in range(X.shape[0]):
        if lr_score(X[i], params) > 0.5:
            errors += (1 - Y[i].item()) / 2
        else:
            errors += (1 + Y[i].item()) / 2
    return 1-errors/X.shape[0]

def lr_score(x, params):
    x.shape = (-1,)
    x_one = np.concatenate((np.array([1]), x))
    z = (x_one @ params).item()
    return 1. / (1 + exp(-z))

def plotPath(plot_path=False, plot_fit=True):
    train = loadtxt('data/data1_train.csv')
    X = train[:,0:2]
    Y = train[:,2:3]
    params, path = lr.train(X, Y, reg='l2', alpha=0., eta=.1, max_iters=1000)
    w_0 = []
    w_1 = []
    w_2 = []
    for point in path:
        w_0.append(point[0][0])
        w_1.append(point[1][0])
        w_2.append(point[2][0])

    print([w_0[-1], w_1[-1], w_2[-1]])
    tag = '(λ = 0)'

    if plot_path:
        iters = np.linspace(0, len(path), len(path))
        plt.plot(iters, w_0, label='w_0')
        plt.plot(iters, w_1, label='w_1')
        plt.plot(iters, w_2, label='w_2')
        plt.title('Batch Gradient Descent '+tag)
        plt.xlabel('Number of Iterations')
        plt.ylabel('Weights')
        plt.legend()
        plt.show()

    if plot_fit:
        validate = loadtxt('data/data1_validate.csv')
        X_val = validate[:,0:2]
        Y_val = validate[:,2:3]
        predictLR = lambda x: lr_score(x, params)
        plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Train '+tag)
        plotDecisionBoundary(X_val, Y_val, predictLR, [0.5], title = 'LR Validate '+tag)
        pl.show()

def model_sweep(plot=False):
    for name in ['1', '2', '3', '4']:
        for reg in ['l1', 'l2']:
            best_tag, best_params, best_train_acc, best_val_acc, best_test_acc = None,None,0,0,0
            for alpha in [0.5 ** i for i in range(100)]:
                # training set
                train = loadtxt('data/data'+name+'_train.csv')
                X = train[:,0:2]
                Y = train[:,2:3]

                # validation set
                validate = loadtxt('data/data'+name+'_validate.csv')
                X_val = validate[:,0:2]
                Y_val = validate[:,2:3]

                # test set
                test = loadtxt('data/data'+name+'_test.csv')
                X_test = test[:,0:2]
                Y_test = test[:,2:3]                

                params = lr.sktrain(X, Y.flatten(), reg='l1', alpha=alpha)

                tag = '(data{name}, {reg} reg, λ = {alpha})'.format(name=name, reg=reg.upper(), alpha=alpha)
                np.set_printoptions(precision=3)

                train_acc = accuracy(X, Y, params)
                val_acc = accuracy(X_val, Y_val, params)
                test_acc = accuracy(X_test, Y_test, params)

                if val_acc > best_val_acc:
                    best_tag = tag
                    best_params = params.flatten()
                    best_train_acc = train_acc
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                
                if plot:
                    predictLR = lambda x: lr_score(x, params)
                    plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Train '+tag)
                    plotDecisionBoundary(X_val, Y_val, predictLR, [0.5], title = 'LR Validate '+tag)
    
        print('description: {tag}'.format(tag=best_tag))
        print('params:', best_params)
        print('training accuracy: {:.2%}'.format(best_train_acc))
        print('validation accuracy: {:.2%}'.format(best_val_acc))
        print('test accuracy: {:.2%}'.format(best_test_acc))
        print('\n')
    if plot:
        pl.show()   

if __name__ == '__main__':
    main()