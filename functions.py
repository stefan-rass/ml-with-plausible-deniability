from numpy.linalg import norm
import numpy as np


def norm_v(x, b):
    """ Projection on the row-space of B, then use semi-norm on the so-projected vector """
    y = b.T @ b @ x
    return 0.5 * semi_norm_b(y, b)


def norm_w1(x, b, w1):
    """ Projection on the 1-dimensional subspace spanned by w1 """
    alpha = semi_norm_b(w1, b)  # this value is constant
    lambda_ = w1.T @ x
    n_y1 = norm(lambda_ * w1, ord=1)
    return 0.5 * alpha * n_y1


def semi_norm_b(x, b):
    """ 2-norm induced by the matrix B """
    return np.sqrt(x.T @ b.T @ b @ x)


def crafted_norm(x, b, w1):
    """ norm from theo  rem 1: b(x) + (norm on V + norm on W1) """
    return semi_norm_b(x, b) + norm_v(x, b) + norm_w1(x, b, w1)


def prepend_ones(x):
    """ prepend a column of ones to x """
    return np.hstack((np.ones((x.shape[0], 1), dtype=int), x))


def conv_b(a, b):
    """ convolve with bias """
    return prepend_ones(a) @ b


def norm_lin_reg_err_bad(p, x, y):
    return norm(p[0] + x@p[1:] - y)


def norm_lin_reg_err(p, x, y):
    """ norm of the linear regression error """
    return norm(conv_b(x, p) - y)


def norm_crafted_lin_reg_err(p, x, y, b, w1):
    """ norm of the crafted linear regression error """
    return crafted_norm(conv_b(x, p) - y, b, w1)


def linear_independence(j, e):
    """ Check if the columns of j and e are linearly independent """
    return np.linalg.matrix_rank(np.hstack((j, e))) == np.linalg.matrix_rank(j),


if __name__ == "__main__":
    import scipy as sp
    import pandas as pd
    from types import SimpleNamespace
    from data import x_training, y_training, p
    X_DIM = 5  # dimension of x (train data)
    initial_guess = np.zeros(X_DIM + 1)

    # show multiple ways of doing linear regression and print for equivalence
    n = SimpleNamespace(orig=p)  # a dict that's settable with setattr
    n.a = sp.optimize.minimize(fun=norm_lin_reg_err_bad, x0=initial_guess, args=(x_training, y_training)).x
    n.b = sp.optimize.minimize(fun=norm_lin_reg_err, x0=initial_guess, args=(x_training, y_training)).x
    n.c = np.linalg.lstsq(a=prepend_ones(x_training), b=y_training, rcond=None)[0]
    print(pd.DataFrame(vars(n)))

