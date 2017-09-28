import pylab as pl
import numpy as np
from math import *
import matplotlib.pyplot as plt


#my code

def gradientDescentOneStep(initial_point, grad_descent_function, step_size):
    gradient_step = grad_descent_function(initial_point)
    return initial_point - step_size*gradient_step
 
def hasConverged(current,prev, calcFunction, threshold, convergence_criteria):
    #calc_function is either an objective function or a gradient function
    converged_yet = True
    error = inf
    if convergence_criteria == "objective":
        objectiveFunction = calcFunction
        error = objectiveFunction(current) - objectiveFunction(prev)
        converged_yet = abs(error) < threshold
    elif convergence_criteria == "norm":
        gradFunction = calcFunction
        error = np.linalg.norm(gradFunction(current))
        converged_yet = np.linalg.norm(gradFunction(current)) < threshold
    else:
        raise ValueError("Invalid convergence criteria")
    return converged_yet, error

def gradientDescent(gradient_fun, start, learn_rate, convergence_specs, step_path):
    
    if isinstance(learn_rate, float):
        rate = lambda x: learn_rate
    else:
        rate = learn_rate
    current = start
    prev = None
    iterations = 0
    maxIters = 20000
    (calcFunction, threshold, convergence_criteria) = convergence_specs
    while prev is None or ( iterations < maxIters):
        converged_yet = False
        error = 0
        if prev is not None:
            converged_yet, error = hasConverged(current, prev, calcFunction,threshold,convergence_criteria)
        if converged_yet:
            break
        prev = current
        iterations += 1
        current = gradientDescentOneStep(current, gradient_fun, rate(iterations))
        step_path.append((iterations, current, error))
        if iterations % 10 == 0:
            pass
            #print(position, iterations)
    return current

def generateGaussian(gMean, gCov):
    mean = np.mat(gMean[:, np.newaxis])
    invCov = np.linalg.inv(gCov)
    normalization = -10000/(sqrt(2*pi)**mean.size * np.linalg.det(gCov))
    def gaussian(x):
        return normalization * exp(-(1/2) *((x - mean).T * invCov * (x - mean)))
    def gaussianGradient(x):
        return -gaussian(x) * invCov * (x - mean)
    return gaussian, gaussianGradient

def generateBowl(qA, qB):
    A = np.mat(qA)
    b = np.mat(qB).T
    def bowl(x):
            return float(.5*x.T * A * x - x.T * b)
    def bowlGradient(x):
        return A * x - b
    return bowl, bowlGradient

def centralDifferences(objectiveFunction, stepSize):
    def gradient(x):
        rows = x.shape[0]
        assert x.size == rows
        directionalGrads = []
        for i in range(rows):
            step = np.zeros_like(x)
            step[i] = stepSize
            #Compute the derivative in each direction
            directionalGrads.append((objectiveFunction(x + step) - objectiveFunction(x - step)) / (2 * stepSize))
        return np.mat(directionalGrads).T
    return gradient


def getData():

    # load the parameters for the negative Gaussian function and quadratic bowl function
    # return a tuple that contains parameters for Gaussian mean, Gaussian covariance,
    # A and b for quadratic bowl in order

    data = pl.loadtxt('parametersp1.txt')

    gaussMean = data[0,:]
    gaussCov = data[1:3,:]

    quadBowlA = data[3:5,:]
    quadBowlb = data[5,:]

    return (gaussMean,gaussCov,quadBowlA,quadBowlb) 



if __name__ == '__main__':
    gaussMean, gaussCov, quadBowlA, quadBowlb = getData()
    
    gauss, gaussGrad = generateGaussian(gaussMean, gaussCov)
    bowl, bowlGrad = generateBowl(quadBowlA, quadBowlb)
    
    
    
    #function, threshold, obj vs. norm
    start = np.mat([[0.], [0.]])
    step_size = .01
    
    #### PLOTS HERE ####
    
    #Central Differences, Gauss
    numStepSizes = 4
    stepSizes = [10**-i for i in range(numStepSizes)]
    numDists = 20
    gradErrs = np.zeros((numStepSizes,numDists),dtype = np.matrix)
    for dist in range(1,numDists):
        start = np.mat([[10.], [10.-dist]])
        truthGrad=gaussGrad(start)
        for i in range(numStepSizes):
            approxFn = centralDifferences(gauss,stepSizes[i])
            gradErrs[i][dist] = np.linalg.norm(approxFn(start) - truthGrad)
    
    for i in range(numStepSizes):
        plt.plot(range(numDists),gradErrs[i],label='-log(Step Size) = '+str(i))
    plt.legend()
    plt.xlabel('Distance from Critical Point')
    plt.ylabel('Norm of Difference From True Gradient')
    plt.title('Central Difference Gradient Analysis (Gauss)')
    plt.show()
    
    
    #Central Differences, Bowl
    numStepSizes = 4
    stepSizes = [10**-i for i in range(numStepSizes)]
    numDists = 20
    gradErrs = np.zeros((numStepSizes,numDists),dtype = np.matrix)
    for dist in range(1,numDists):
        start = np.mat([[40/3.], [40/3.-dist]])
        truthGrad=bowlGrad(start)
        for i in range(numStepSizes):
            approxFn = centralDifferences(bowl,stepSizes[i])
            gradErrs[i][dist] = np.linalg.norm(approxFn(start) - truthGrad)
    
    for i in range(numStepSizes):
        plt.plot(range(numDists),gradErrs[i],label='-log(Step Size) = '+str(i))
    plt.legend()
    plt.xlabel('Distance from Critical Point')
    plt.ylabel('Norm of Difference From True Gradient')
    plt.title('Central Difference Gradient Analysis (Bowl)')
    plt.show()
    
    #Norm of Gradient, Gauss
    convergence_specs = (gauss,1e-10,"objective")
    start = np.mat([[10], [10-10]])
    step_size = .01
    steps = []
    result = gradientDescent(gaussGrad, start, step_size, convergence_specs, steps)
    
    plt.plot(range(len(steps)),[i[2] for i in steps])
    plt.xlabel("Steps")
    plt.ylabel("Gradient Norm")
    plt.title("Varying Convergence Threshold (Gauss)")
    plt.show()
    
    #Norm of Gradient, bowl
    convergence_specs = (bowl,1e-10,"objective")
    start = np.mat([[10], [10-10]])
    step_size = .01
    steps = []
    result = gradientDescent(bowlGrad, start, step_size, convergence_specs, steps)
    
    plt.plot(range(len(steps)),[i[2] for i in steps])
    plt.xlabel("Steps")
    plt.ylabel("Gradient Norm")
    plt.title("Varying Convergence Threshold (Gauss)")
    plt.show()
    
    

    #Distance from critical point, Gaussian
    convergence_specs = (gauss,1e-10,"objective")
    numDists = 13
    stepsToConvergence = [0]*numDists
    for i in range(numDists):
        start = np.mat([[10.], [10.-i]])
        steps = []
        result = gradientDescent(gaussGrad, start, step_size, convergence_specs, steps)
        stepsToConvergence[i] = len(steps)
    
    plt.scatter(range(numDists),stepsToConvergence)
    plt.xlabel("Starting Distance From Critical Point")
    plt.ylabel("Number of Descent Steps to Convergence")
    plt.title("Varying Starting Point (Gaussian)")
    plt.show()
    
    #Distance from critical point, Bowl
    convergence_specs = (bowl,1e-10,"objective")
    numDists = 30
    stepsToConvergence = [0]*numDists
    for i in range(numDists):
        start = np.mat([[40./3], [40./3.-i]])
        steps = []
        result = gradientDescent(bowlGrad, start, step_size, convergence_specs, steps)
        stepsToConvergence[i] = len(steps)
    
    plt.scatter(range(numDists),stepsToConvergence)
    plt.xlabel("Starting Distance From Critical Point")
    plt.ylabel("Number of Descent Steps to Convergence")
    plt.title("Varying Starting Point (Bowl)")
    plt.show()
    
    #Stepsize, Gauss
    convergence_specs = (gauss,1e-10,"objective")
    numSteps = 6
    stepsToConvergence = [0]*numSteps
    start = np.mat([[10], [10-10]])
    stepSizes = [(i)/2+1. for i in range(numSteps)]
    for i in range(numSteps):
        step_size = 10**(-stepSizes[i])
        steps = []
        result = gradientDescent(gaussGrad, start, step_size, convergence_specs, steps)
        stepsToConvergence[i] = len(steps)
    
    plt.scatter(stepSizes,stepsToConvergence)
    plt.xlabel("-log(Step Size)")
    plt.ylabel("Number of Descent Steps to Convergence")
    plt.title("Varying Step Size (Gauss)")
    plt.show()
    
    #Stepsize, Bowl
    convergence_specs = (bowl,1e-10,"objective")
    numSteps = 6
    stepsToConvergence = [0]*numSteps
    start = np.mat([[40./3], [40./3-10]])
    stepSizes = [(i)/2+1. for i in range(numSteps)]
    for i in range(numSteps):
        step_size = 10**(-stepSizes[i])
        steps = []
        result = gradientDescent(bowlGrad, start, step_size, convergence_specs, steps)
        stepsToConvergence[i] = len(steps)
    
    plt.scatter(stepSizes,stepsToConvergence)
    plt.xlabel("-log(Step Size)")
    plt.ylabel("Number of Descent Steps to Convergence")
    plt.title("Varying Step Size (Bowl)")
    plt.show()
    
    
    
    #Convergence Criteria, Gauss
    convergence_specs = (gauss,1e-10,"objective")
    numSteps = 15
    stepsToConvergence = [0]*numSteps
    start = np.mat([[10], [10-10]])
    step_size = .01
    convergences = [10**-i for i in range(numSteps)]
    for i in range(numSteps):
        convergence_specs = (gauss,convergences[i],"objective")
        steps = []
        result = gradientDescent(gaussGrad, start, step_size, convergence_specs, steps)
        stepsToConvergence[i] = len(steps)
    
    plt.scatter(range(numSteps),stepsToConvergence)
    plt.xlabel("-log(Convergence Threshold)")
    plt.ylabel("Number of Descent Steps to Convergence")
    plt.title("Varying Convergence Threshold (Gauss)")
    plt.show()
    
    #Convergence Criteria, Bowl
    convergence_specs = (bowl,1e-10,"objective")
    numSteps = 15
    stepsToConvergence = [0]*numSteps
    start = np.mat([[40./3], [40./3-10]])
    step_size = .01
    convergences = [10**-i for i in range(numSteps)]
    for i in range(numSteps):
        convergence_specs = (bowl,convergences[i],"objective")
        steps = []
        result = gradientDescent(bowlGrad, start, step_size, convergence_specs, steps)
        stepsToConvergence[i] = len(steps)
    
    plt.scatter(range(numSteps),stepsToConvergence)
    plt.xlabel("-log(Convergence Threshold)")
    plt.ylabel("Number of Descent Steps to Convergence")
    plt.title("Varying Convergence Threshold (Bowl)")
    plt.show()
    
    
    
    
    
    
        
    #print(result[0],result[1])
    #plt.plot(float(result[0]), float(result[1]), 'bo')
#     for i in range(len(steps)):
#         
#         if i%100 == 0:
#             #print(steps[i][2])
#             plt.plot(i,steps[i][2],'bo')
#             xx = steps[i][1]
#             #plt.plot(xx[0],xx[1],'rx')
#             
#     print('err ends')
        
    #plt.show()
    #print(steps)
    #rn = 500
    #xs = [(i-rn/2.)/10 for i in range(rn)]
    #ys = [gauss(x) for x in xs]
    #plt.plot(xs,ys)
    #print(steps[-1])
    #print(quadBowlA,quadBowlb)
    #plt.show()

    #print([gaussGrad(i/10) for i in range(10)])
    
#     for i in range(10):
#         stepSize = 10**-i
#         approxGaussGrad = centralDifferences(gauss, stepSize)
#         start_point = np.mat((8.,9.)).T
#         #print(approxGaussGrad(start_point))
#         print(stepSize, np.linalg.norm(gaussGrad(start_point) - approxGaussGrad(start_point)))
    
    
# (gMean,gCov,qA,qb) = getData()
# (g1,g2) = generateGaussian(gMean,gCov)
# 
# rn = 500
# xs = [(i-rn/2.)/10 for i in range(rn)]
# ys = [g1(x) for x in xs]
# 
# print(gMean,gCov)
# plt.plot(xs,ys)
# plt.show()