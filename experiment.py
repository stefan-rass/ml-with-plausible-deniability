import numpy as np
import pandas as pd
import scipy as sp
from numpy.linalg import matrix_rank

from functions import norm_crafted_lin_reg_err, prepend_ones, conv_b, linear_independence

NR_RECORDS = 10  # number of train data records
X_DIM = 5  # dimension of x (train data)
SEED = False  # 0/False -> use the values from the execution_snapshot

# wide dataframes in console for nicer prints
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option("expand_frame_repr", False)


if SEED:
    gen = np.random.Generator(np.random.PCG64(SEED))
    x_training = gen.random((NR_RECORDS, X_DIM))
    p_true = 6 * gen.random(X_DIM + 1) - 3
    y_training = conv_b(x_training, p_true) + -1 / 5 * np.log(gen.random(NR_RECORDS))
    x_decoy = gen.random((NR_RECORDS, X_DIM))
    y_decoy = gen.random(NR_RECORDS)
    e = p = B = p2 = None
    rand_mul = gen.random(1)
else:  # no seed -> use the data from the original experiment and compare the results
    from data import B, e, p2, p_true, rand_mul, x_decoy, x_training, y_decoy, y_training


initial_guess = np.zeros(X_DIM + 1)
# three ways of doing linear regression
p1 = np.linalg.lstsq(a=prepend_ones(x_training), b=y_training, rcond=None)[0]


e_ = y_decoy - conv_b(x_decoy, p1)
e_ = np.expand_dims(e_, axis=1)
j = prepend_ones(x_training)
assert linear_independence(j, e_)

print('np.linalg.matrix_rank condition OK; constructing the norm')
b = np.linalg.pinv(sp.linalg.null_space(e_.T))  # moore-penrose inverse of the null space of w0

assert B is None or np.allclose(b, B, atol=1)
assert e is None or np.allclose(e_.T, e, atol=1)

w = e_.copy()
w[0] += 1  # small distortion of e should be enough
while matrix_rank(np.hstack((w, e_))) < 2 or matrix_rank(b) >= matrix_rank(np.hstack((b.T, w)).T):
    w = np.random.default_rng(SEED).random(*e_.shape)  # retry with a random vector (should succeed with probability 1)
w = w / np.sum(w)  # scale to unit length


print('re-fitting')
x_noise = p1 + 0.1 * rand_mul
p_crafted = sp.optimize.minimize(fun=norm_crafted_lin_reg_err, x0=x_noise, args=(x_decoy, y_decoy, b, w)).x
assert p2 is None or np.allclose(p_crafted, p2, atol=1)

print(pd.DataFrame({"p_true": p_true, "p_crafted": p_crafted, "p1": p1}), "\n")
y_train_rec_orig = conv_b(x_training, p1)
y_train_rec_fit = conv_b(x_training, p_crafted)
print(pd.DataFrame({"y_train": y_training, "y_train_rec_orig": y_train_rec_orig, "y_train_rec_fit": y_train_rec_fit}), "\n")
y_decoy_rec_orig = conv_b(x_decoy, p1)
y_decoy_rec_fit = conv_b(x_decoy, p_crafted)
print(pd.DataFrame({"y_decoy": y_decoy, "y_decoy_rec_orig": y_decoy_rec_orig, "y_decoy_rec_fit": y_decoy_rec_fit}), "\n")

# check if reconstructed with original and crafted are close
assert np.allclose(y_train_rec_orig, y_train_rec_fit, atol=0.01)
assert np.allclose(y_decoy_rec_orig, y_decoy_rec_fit, atol=0.01)
