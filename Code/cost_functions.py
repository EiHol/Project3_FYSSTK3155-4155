"""
This file contains various cost functions and some of their derivatives used in machine learning.
"""

# Import libraries
import autograd.numpy as np

# Mean squared error function
def mse(predict, target):
    return np.mean((predict - target) ** 2)

# Derivative of the mse function
def mse_der(predict, target):
    return (predict - target) * (2/len(target))

# Mean squared error L1 norm
def mse_L1(predict, target, lmbda):
    pass

# Mean squared error L2 norm
def mse_L2(predict, target, lmbda):
    pass

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
