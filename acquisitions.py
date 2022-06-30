import numpy as np
from skopt.learning import GaussianProcessRegressor
from scipy.stats import norm
from utils import GP_pred


'''
Each acquisition function uses the following set of parameters:
    :param X:           Input dataset, (n,d) array
    :param GP_model:    GP regressor model
    :param fX_best:     Running best evaluated point
    :param fX_best_min: List of minimum running best evaluated points
    :param n:           Experiment number
    :param xi:          EI signal hyperparameter
    :param beta:        LCB exploration ratio hyperparameter
    :param beta_ab:     EI Abrupt exploration ratio hyperparameter
    :param eta:         EI Abrupt learning threshold hyperparameter
    :param ratio:       LCB Adaptive exploration ratio hyperparameter
    :param decay:       LCB Adaptive exponential decay hyperparameter
'''

def EI(X, GP_model, fX_best, xi, fX_best_min=None, beta=None, eta=None, n=None, ratio=None, decay=None, beta_ab=None):
    '''
    Expected Improvement acquisition function.
    :return: acquisition value of the next point to sample in the search space
    '''
    mean, std = GP_pred(X, GP_model)
    z = (fX_best - mean - xi) / std
    return (fX_best - mean - xi) * norm.cdf(z) + std * norm.pdf(z)


def LCB(X, GP_model, beta, fX_best=None, fX_best_min=None, xi=None, eta=None, n=None, ratio=None, decay=None, beta_ab=None):
    '''
    Lower Confidence Bound acquisition function.
    :return: acquisition value of the next point to sample in the search space
    '''
    mean, std = GP_pred(X, GP_model)
    return - mean + beta * std


def LCB_ada(X, GP_model, ratio, decay, n, fX_best=None, fX_best_min=None, xi=None, beta=None, eta=None, beta_ab=None):
    '''
    Lower Confidence Bound Adaptive acquisition function.
    :return: acquisition value of the next point to sample in the search space
    '''
    mean, std = GP_pred(X, GP_model)
    return - mean + ratio * std * np.power(decay, n)

def EI_abrupt(X, GP_model, fX_best, fX_best_min, xi, beta_ab, eta, n, ratio = None, decay = None, beta = None):
    '''
    Expected Improvement Abrupt acquisition function.
    :return: acquisition value of the next point to sample in the search space
    '''
    ac_value = EI(X, GP_model, fX_best, xi)
    if n > 5:
        if abs(min(np.gradient(fX_best_min[-3:]))) > eta: # activate EI abrupt
            ac_value = EI(X, GP_model, fX_best, xi)
        else:
            ac_value = LCB(X, GP_model, beta_ab)
    return ac_value