import pdb
import random
import pylab as pl
import matplotlib.pyplot as plt

import numpy as np
import scipy.linalg as linalg 
import P2.loadFittingDataP2 as lf2


# X is an array of N data points (one dimensional for now), that is, NX1
# Y is a Nx1 column vector of data values

def getData(name):
    data = pl.loadtxt(name)
    # Returns column matrices
    X = data[0:1].T
    Y = data[1:2].T
    return X, Y

def regressAData():
    return getData('regressA_train.txt')

def regressBData():
    return getData('regressB_train.txt')

def validateData():
    return getData('regress_validate.txt')

def createFeatures(X=np.zeros(1), M=1):
    features = np.ones_like(X)
    for i in range(1, M+1):
        features = np.hstack([features, np.power(X,i)])
    return features


def ridgeRegression(X, Y, l=0, M=1, features=False):
    if not features:
        features = createFeatures(X, M)
    else:
        features = X
        M = features.shape[1]-1

    firstTerm = linalg.pinv(np.dot(np.transpose(features), features) + l*np.identity(M+1))
    secondTerm = np.transpose(features)
    thirdTerm = Y
    theta = np.dot(firstTerm, np.dot(secondTerm, thirdTerm))
    linear_error = np.linalg.norm(np.dot(features,theta) - Y)**2
    regularization_error = np.linalg.norm(l*theta)**2

    return theta, (linear_error+regularization_error)/float(Y.size)

def predictRidge(X, theta, features=False):
    if not features:
        features = createFeatures(X, M=theta.size-1)
    else:
        features = X
    
    predictions = np.dot(features, theta)

    return predictions


def MSE(predictions, actuals):
    mse = np.dot(np.transpose(predictions-actuals), (predictions-actuals))/float(predictions.size)

    return mse 

if __name__ == '__main__':

    X,Y = lf2.getData(False)
    #print(Y.shape)
    Xvals = np.matrix(X).T
    Yvals = np.matrix(Y).T
    
#     dim_rn = range(5)
#     lambda_rn = [i/2 for i in range(5)]
#     for i in lambda_rn:
#         errs = [0]*len(dim_rn)
#         for j in dim_rn:
#             theta, error = ridgeRegression(Xvals, Yvals, l=i, M=j)
#             errs[j] = error
#         plt.plot(dim_rn,errs,label = "lambda = "+str(i))
#      
#     plt.xlabel("Dimension")
#     plt.ylabel("Mean Squared Errors")
#     plt.title("SSE vs. Regression Dimensionality for Polynomial Basis")
#     plt.legend()
#     plt.show()
#     
    #print(type(theta))

    
    AData_X,AData_Y = regressAData()
    BData_X,BData_Y = regressBData()
    valData_X,valData_Y = validateData()
    
    Ms = [1,2,3,4,5,6,7]
    ls = [0,.5,1,1.5,2,2.5,3]
    
    #Rows are lambda, columns are M
    thetas = np.zeros((len(ls),len(Ms)),dtype=np.matrix)
    MSEs = np.zeros((len(ls),len(Ms)))
    
    #Switch these around as needed
    train_set_X,train_set_Y = regressAData()
    test_set_X,test_set_Y = regressBData()
    
    for j in range(len(Ms)):
        for i in range(len(ls)):
            M_j = Ms[j]
            l_i= ls[i]
            thetas[i][j],temp = ridgeRegression(train_set_X, train_set_Y, l=l_i, M=M_j)
            MSEs[i][j] = MSE(predictRidge(valData_X,thetas[i][j]),valData_Y)
    

    print("MSEs: ",MSEs)
    bestModel = np.unravel_index(MSEs.argmin(),MSEs.shape)
    bestLambda = ls[bestModel[0]]
    bestM = Ms[bestModel[1]]
    print("Best Lambda, Best M: ",(bestLambda,bestM))
    bestTheta = thetas[bestModel[0]][bestModel[1]]
    print(bestTheta)
    #Have to find best M for each lambda

    print("prediction:",predictRidge(test_set_X,bestTheta))
    test_error = MSE(predictRidge(test_set_X,bestTheta),test_set_Y)
    print("Train error: ",MSEs[bestModel[0]][bestModel[1]])
    print("Test error: ",test_error)
    
    fig = plt.figure(figsize=(6, 3.2))

    ax = fig.add_subplot(111)
    ax.set_title('MSEs on Validation Set')
    plt.imshow(MSEs)
    ax.set_aspect('equal')
    
    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical')
    plt.show()
    

#print(theta)

