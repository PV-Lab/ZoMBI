# def warn(*args, **kwargs):
#     pass
# import warnings
# warnings.warn = warn
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import time
import random
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
from utils import bounded_LHS, progress_bar, plot
from acquisitions import *

class ZoMBI:
    def __init__(self, dataset_X, dataset_fX, fX_model, BO, nregular, activations, memory, forward, ensemble,
                 xi=0.1, beta=1, beta_ab=0.1, eta=0, ratio=3, decay=0.9):
        '''
        Runs the ZoMBI optimization procedure over #activations after #nregular standard BO.
        :param dataset_X:       An (n,d) array of points to run BO on
        :param dataset_fX:      An (n,) array of labels for the respective X values
        :param fX_model:        Predictive model to obtain f(X) from X
        :param BO:              BO acquisition function: LCB, LCB_ada, EI, EI_abrupt
        :param nregular:        Number of experiments to run regular BO before ZoMBI
        :param activations:     Number of times to activate ZoMBI optimization
        :param memory:          Number of points to retain in ZoMBI memory
        :param forward:         Number of forward experiments to run per ZoMBI activation
        :param ensemble:        Number of independent BO/ZoMBI runs
        :param xi:              EI signal hyperparameter
        :param beta:            LCB exploration ratio hyperparameter
        :param beta_ab:         EI Abrupt exploration ratio hyperparameter
        :param eta:             EI Abrupt learning threshold hyperparameter
        :param ratio:           LCB Adaptive exploration ratio hyperparameter
        :param decay:           LCB Adaptive exponential decay hyperparameter
        '''

        if not (BO == LCB or BO == EI or BO == LCB_ada or BO == EI_abrupt):
            raise ValueError("Variable BO must be set to one of the following objects: LCB, EI, LCB_ada, EI_abrupt")
        if not (forward > 1):
            raise ValueError("Number of forward experiments must be greater than 1")

        # Input parameters
        self.dataset_X = dataset_X
        self.dataset_fX = dataset_fX
        self.fX_model = fX_model
        self.BO = BO                    # BO acquisition function to use
        self.nregular = nregular        # number of regular BO experiments to run before ZoMBI
        self.activations = activations  # number of ZoMBI activations
        self.memory = memory            # number of best memory points to keep
        self.forward = forward          # number of forward experiments per activation
        self.ensemble = ensemble        # number of independent ensemble model runs
        self.batch = 1                  # batch size
        self.compute = []

        # Acquisition function parameters
        self.xi = xi
        self.beta = beta
        self.beta_ab = beta_ab
        self.eta = eta
        self.ratio = ratio
        self.decay = decay

        # GP model
        self.GP = GaussianProcessRegressor(kernel = ConstantKernel(1, constant_value_bounds = 'fixed') * Matern(
            length_scale = 5, length_scale_bounds = 'fixed', nu = 1) * 0.5, n_restarts_optimizer = 30,
            alpha = 0.0002, normalize_y = False)

    def activate_zombi(self, current_act, total_act):
        '''
        Activate ZoMBI when called.
        :param current_act:     The current activation number
        :param total_act:       The total number of activations in the procedure
        :return:                Optimized X values, Y values, minimum Y values
        '''

        idx_old = np.array([0])
        mem = np.unique(self.fX_i, return_index=True)[1][-self.memory:]
        min_vector = np.min(self.X_i[mem, :], axis=0)  # min bounds of last sampled points
        max_vector = np.max(self.X_i[mem, :], axis=0)  # max bounds of last samples points
        if current_act == 0:
            self.bound_l_i = min_vector
            self.bound_u_i = max_vector
        else:
            self.bound_l_i = np.vstack([self.bound_l_i, min_vector])
            self.bound_u_i = np.vstack([self.bound_u_i, max_vector])
        X_bounded = bounded_LHS(N=self.memory, min_vector=min_vector, max_vector=max_vector)  # bounded initialization dataset from LHS
        fX_bounded = -1 * self.fX_model(X_bounded)
        bounded_norm = self.X_i.copy()
        for d in range(bounded_norm.shape[1]):
            bounded_norm[bounded_norm[:, d] > max_vector[d], d] = max_vector[d]  # upper bound
            bounded_norm[bounded_norm[:, d] < min_vector[d], d] = min_vector[d]  # lower bound
        inc = 0.1
        for n in range(self.forward):
            inc = progress_bar(n = n, T = self.forward, inc = inc, ensemble = self.e,
                               text = f'ZoMBI Activation {current_act + 1} / {total_act}')
            X_new = X_bounded
            fX_new = fX_bounded
            start = time.time()
            self.GP.fit(X_new, fX_new) # fit a zoomed-in, GP resolved near optimum
            ac_value = self.BO(X=bounded_norm, GP_model=self.GP, n=n, fX_best=self.fX_min_i[-1], fX_best_min=self.fX_min_i,
                               xi=self.xi, beta=self.beta, beta_ab=self.beta_ab, eta=self.eta,
                               ratio=self.ratio, decay=self.decay)  # apply zooming constraints to search space
            self.compute_i.append(time.time()-start)
            idx = np.argsort(ac_value)[:self.batch]
            j = 0
            if idx == idx_old:
                j = j + 1
                idx = np.argsort(ac_value)[j + 1:self.batch + j + 1]
            X_bounded = np.vstack([X_new, bounded_norm[idx]])
            fX_bounded_new = -1 * self.fX_model(bounded_norm[idx]) # evaluate target function at the optimum X
            fX_bounded = np.append(fX_new, fX_bounded_new)
            # fX_bounded = np.append(fX_new, self.dataset_fX[idx]) # original implementation
            
            idx_old = idx

        X_bounded_full = np.array(X_bounded)
        fX_bounded_full = np.array(fX_bounded)
        fX_bounded_min = np.array(np.minimum.accumulate(-1 * fX_bounded))
        return X_bounded_full, fX_bounded_full, fX_bounded_min

    def optimize(self, X_initial, fX_initial, plot_f = True):
        '''
        Executes the full BO/ZoMBI optimization procedure when called.
        :param X_initial:       An (n,d) array of initialization points, best to use LHS with n~5
        :param fX_initial:      An (n,) array of evaluated initilization points f(X)
        :param plot_f:          True or False value, plots the function value, compute time, and bounds evolution
        :return:                Call func.X, func.fX to get the optimized
        '''

        for e in range(self.ensemble):
            self.e = e
            fX_best_list = []
            idx_old = np.array([0])
            inc = 0.1
            X_initial_i = X_initial
            fX_initial_i = fX_initial
            self.compute_i = []


            for n in range(self.nregular): # Regular BO
                inc = progress_bar(n = n, T = self.nregular, inc = inc, ensemble = self.e,
                                   text = f'Standard BO ({self.BO.__name__})')
                X_new = X_initial_i
                fX_new = fX_initial_i
                fX_best = max(fX_new)
                fX_best_list.append(fX_best)
                fX_best_min = np.minimum.accumulate(-1 * np.array(fX_best_list))
                start = time.time()
                self.GP.fit(X_new, fX_new)
                ac_value = self.BO(X=X_new, GP_model=self.GP, n=n, fX_best=fX_best, fX_best_min=fX_best_min,
                                   xi=self.xi, beta=self.beta, beta_ab=self.beta_ab, eta=self.eta,
                                   ratio=self.ratio, decay=self.decay)
                self.compute_i.append(time.time()-start)
                idx = np.argsort(ac_value)[:self.batch]
                j = 0
                if idx == idx_old:
                    j = j + 1
                    idx = np.argsort(ac_value)[j + 1:self.batch + j + 1]
                X_initial_i = np.vstack([X_new, self.dataset_X[idx]])  # Grab predicted x-value from dataset
                fX_initial_i = np.append(fX_new, self.dataset_fX[idx])  # Grab predicted y-value from dataset
                idx_old = idx
            self.X_i = X_initial_i
            self.fX_i = fX_initial_i
            self.fX_min_i = np.minimum.accumulate(-1 * self.fX_i)

            for a in range(self.activations):
                X_i, fX_i, fX_min_i = self.activate_zombi(current_act = a, total_act = self.activations)
                self.X_i = np.vstack([self.X_i, X_i])
                self.fX_i = np.hstack([self.fX_i, fX_i])
                self.fX_min_i = np.hstack([self.fX_min_i, fX_min_i])
            self.fX_min_i = np.minimum.accumulate(self.fX_min_i)

            if e == 0:
                self.X = self.X_i
                self.fX = self.fX_i
                self.fX_min = self.fX_min_i
                self.compute = np.array(self.compute_i)
                self.bound_l = self.bound_l_i
                self.bound_u = self.bound_u_i
            else:
                self.X = np.vstack([self.X, self.X_i])
                self.fX = np.vstack([self.fX, self.fX_i])
                self.fX_min = np.vstack([self.fX_min, self.fX_min_i])
                self.compute = np.vstack([self.compute, np.array(self.compute_i)])
                self.bound_l = np.hstack([self.bound_l, self.bound_l_i])
                self.bound_u = np.hstack([self.bound_u, self.bound_u_i])

        if plot_f == True:
            plot(fX_min = self.fX_min, compute = self.compute, nregular = self.nregular,
             bound_l = self.bound_l, bound_u = self.bound_u, dim = X_i.shape[1], ensemble = self.ensemble, activations = self.activations)