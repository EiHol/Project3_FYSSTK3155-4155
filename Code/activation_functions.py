"""
This file contains various activation functions and some of their derivatives used in machine learning
"""

# Import libraries
import autograd.numpy as np

# ReLU function
def ReLU(z):
    return np.where(z > 0, z, 0)

# Derivative of the ReLU function
def ReLU_der(z):
    return np.where(z > 0, 1, 0)

# Leaky ReLU function?
def leaky_ReLU(z, a):
    return np.where(z > 0, z, z * a)

# Derivative of the leaky ReLU function?
def leaky_ReLU_der(z, a):
    return np.where(z > 0, 1, a)

def linear(z):
    return z

# Sigmooid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Derivative of the sigmoid function
def sigmoid_der(z):
    return (np.exp(-z) / (1 + np.exp(-z))**2)

# Softmax function
def softmax(z):
    e_z = np.exp(z - np.max(z, axis=0))
    return e_z / np.sum(e_z, axis=1)[:, np.newaxis]

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
