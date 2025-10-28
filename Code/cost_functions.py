"""
This file contains various cost functions and some of their derivatives used in machine learning.
"""

# Import libraries
import autograd.numpy as np

# Mean Squared Error with optional L1 and L2 regularization
def mse(y_true, y_pred, weights=None, l1_lambda=0, l2_lambda=0):
    # Mean squared error
    mse = np.mean((y_true - y_pred) ** 2)

    # Regularization terms
    l1_penalty = 0
    l2_penalty = 0
    if weights is not None:
        l1_term = l1_lambda * np.sum(np.abs(weights))
        l2_term = l2_lambda * np.sum(weights ** 2)

    return mse + l1_penalty + l2_penalty


# Derivative of  Mean Squared Error with optional L1 and L2 regularization
def mse_der(y_true, y_pred, weights=None, l1_lambda=0.0, l2_lambda=0.0):
    # Derivative w.r.t predictions, where a = y_pred
    dC_da = 2 * (y_pred - y_true) / y_true.size

    # Derivative w.r.t weights - regularization only
    dC_dw = None
    if weights is not None:
        # sign(weights) gives the subgradient for L1
        dC_dw = l1_lambda * np.sign(weights) + 2 * l2_lambda * weights

    return dC_da, dC_dw

# Binary cross entropy
def binary_cross_entropy():
    pass

# Binary cross entropy with L1 norm
def BCE_L1():
    pass

# Binary cross entropy with L2 norm
def BCE_L2():
    pass

# Cross entropy function
def cross_entropy(predict, target):
    return np.sum(-target * np.log(predict))

# Softmax cross entropy derivative
def cross_entropy_der(predict, target):
    return predict - target

# Multiclass cross entropy function
def multi_cross_entropy(predict, target):
    pass
