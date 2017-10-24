import numpy as np
import gradientDescent as gd
import logisticRegression as lr

class TestGD:
    def test_batch(self):
        x = np.array([[-10]])
        obj = lambda x: (x@x).item()
        grad = gd.num_grad_fn(obj)
        np.testing.assert_almost_equal(gd.batch(grad, x), 0)

    def test_sgd(self):
        x = np.array([[-10]])
        obj = lambda x: (x@x).item()
        grad = gd.num_grad_fn(obj)
        grad_gen = lambda x: grad
        np.testing.assert_almost_equal(gd.sgd(grad_gen, x), 0)

def main():
    print("Success!")

if __name__ == '__main__':
    main()