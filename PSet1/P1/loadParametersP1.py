import pylab as pl

#my code

def gradientDescentOneStep(initial_point, grad_descent_function, step_size):
    gradient_step = grad_descent_function(initial_point)
    return initial_point - step_size*gradient_step

def hasConverged(point, threshold, max_iters):
    pass

def gradientDescent(gradient_fun, start, rate, isConverged, data):
    pass

def generateGaussian(mean, cov):
    pass

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

