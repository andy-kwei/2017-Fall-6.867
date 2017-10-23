import numpy as np

def num_grad_fn(obj_fn, step_size=1e-8):
    def gradient(x):
        rows = x.shape[0]
        grad = np.zeros((rows, 1))
        for i in range(rows):
            delta = np.zeros((rows, 1))
            delta[i][0] = step_size
            grad[i][0] = (obj_fn(x+delta) - obj_fn(x-delta)) / (2*step_size)
        return grad
    return gradient

def batch(grad_fn, start, eta, epsilon=1e-8, cvg_test=None, max_iters=1e3):
    iters = 0
    if isinstance(eta, float):
        num = eta
        eta = lambda x: num

    prev = start
    curr = start - grad_fn(start) * eta(iters)
    path = [prev, curr]

    if cvg_test == None:
        cvg_test = lambda x, y: np.linalg.norm(grad_fn(y))

    while cvg_test(prev, curr) > epsilon:
        iters += 1
        if iters > max_iters:
            print("Max iterations exceeded.")
            break
        prev = curr
        curr = prev - grad_fn(prev) * eta(iters)
        path.append(curr)
    return curr, path

def sgd(grad_fn_generator, start, eta, epsilon=1e-8, cvg_test=None, max_iters=1e5):
    iters = 0
    if isinstance(eta, float):
        num = eta
        eta = lambda x: num

    grad_fn = grad_fn_generator(0)
    prev = start
    curr = start - grad_fn(start) * eta(0)
    path = [prev, curr]

    if cvg_test == None:
        cvg_test = lambda x, y: np.linalg.norm(grad_fn(y))

    while cvg_test(prev, curr) > epsilon:
        iters += 1
        if iters > max_iters:
            print("Max iterations exceeded.")
            break
        grad_fn = grad_fn_generator(iters)
        prev = curr
        curr = prev - grad_fn(prev) * eta(iters)
        path.append(curr)
    return curr, path