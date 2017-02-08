import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


def linear(x):
    return 0.5*x
