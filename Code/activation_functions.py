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

# Leaky ReLU function
def leaky_ReLU(z, a):
    return np.where(z > 0, z, z * a) # comment

# Derivative of the leaky ReLU function?
def leaky_ReLU_der(z, a):
    return np.where(z > 0, 1, a)

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
