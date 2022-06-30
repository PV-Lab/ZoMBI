import numpy as np
import sys
import matplotlib.pyplot as plt

def GP_pred(X, GP_model):
    '''
    Predict f(X) means and standard deviations from data using GP.
    :param X:           Input dataset, (n,d) array
    :param GP_model:    GP regressor model
    :return:            Predicted posterior means and standard deviations
    '''
    mean, std = GP_model.predict(X, return_std=True)
    return mean, std

def bounded_LHS(N, min_vector, max_vector):
    '''
    Runs bounded Latin Hypercube Sampling (LHS) using the iteratively zoomed-in search space bounds and memory points.
    Used to initialize the start of the forward experiments within each ZoMBI activation.
    :param N:               Number of LHS datapoints to output
    :param min_vector:      A (d,) array of lower bounds
    :param max_vector:      A (d,) array of upper bounds
    :return:                An (N,d) array of X datapoints
    '''

    samples = []
    for n in range(N):
    # randomly select points using uniform sampling within limit ranges
        samples.append(np.random.uniform(low = min_vector, high = max_vector).T)
    return np.vstack(samples)

def progress_bar(n, T, inc, ensemble, text = ''):
    '''
    Progress bar for optimization procedure.
    :param n:           The current experiment number
    :param T:           The total number of experiments to run
    :param inc:         Increment period for updating progress bar, 0.1/1 => 10 updates
    :param ensemble:    The current ensemble number
    :param text:        Text to display either standard BO or ZoMBI with activation function name
    :return:            Live updating progress bar
    '''

    p = round(n/T, 2)
    total = 40
    step = int(p*total)
    block = '\u275A'
    space = '\u205F\u205F'
    if p >= inc:
        sys.stdout.write(f"\r{'Ensemble '+str(ensemble + 1)+', '+text+' : ['+block*step + space*(total-step)+'] '}{n + 1} / {T} experiments completed . . .{' '*step}")
        sys.stdout.flush()
        inc += 0.1
    elif (1 - inc) < 0.1 and inc <= 1:
        step = int(total)
        sys.stdout.write(f"\r{'Ensemble '+str(ensemble + 1)+', '+text+' : ['+block*step + space*(total-step)+']'} {T} / {T} Complete!{' '*step}")
        sys.stdout.flush()
        inc += 0.1
    return inc

def plot(fX_min, compute, nregular, bound_l, bound_u, dim, ensemble, activations):
    '''
    Plots the results of the ZoMBI optimization procedure, once completed.
    :param fX_min:      An (e,n) array of evaluated fX values, where e is the #ensemble
    :param compute:     An (e,n) array of compute times of the algorithm, where e is the #ensemble
    :param nregular:    The number of regular BO experiments to run
    :param bound_l:     A (f,e*d) array of lower bound values, where f is #forward, e is #ensemble, d is #dimensions
    :param bound_u:     A (f,e*d) array of upper bound values, where f is #forward, e is #ensemble, d is #dimensions
    :param dim:         The number of dimensions of the dataset
    :param ensemble:    The number of model ensemble runs
    :param activations: The number of ZoMBI activations
    :return:            Three plots: running minimum fX, compute time, evolution of bound values
    '''
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 20})
    plt.rcParams["axes.linewidth"] = 2.
    plt.rcParams["axes.edgecolor"] = "#505050"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    if len(fX_min.shape) == 1:
        fX_min = fX_min.reshape(fX_min.shape[0], 1).T
        compute = compute.reshape(compute.shape[0], 1).T
    fX_min50 = np.percentile(fX_min, q = 50, axis = 0)
    fX_min10 = np.percentile(fX_min, q = 10, axis = 0)
    fX_min90 = np.percentile(fX_min, q = 90, axis = 0)
    xs = np.linspace(1, len(fX_min50) + 1, len(fX_min50))
    ax1.plot(xs[:nregular], fX_min50[:nregular], lw=3, c='orange', marker='o', markersize=5, alpha=0.8, label='Standard')
    ax1.plot(xs[nregular:], fX_min50[nregular:], lw=3, c='b', marker='o', markersize=5, alpha=0.8, label='ZoMBI')
    ax1.fill_between(xs[:nregular], fX_min10[:nregular], fX_min90[:nregular], color = 'orange', alpha = 0.2)
    ax1.fill_between(xs[nregular:], fX_min10[nregular:], fX_min90[nregular:], color = 'b', alpha=0.2)
    ax1.axvline(nregular, lw=4, c='k', ls=':')
    lmin = np.min(fX_min)
    lmax = np.max(fX_min)
    xpos = nregular / len(fX_min50)
    ax1.text(xpos, 1.05, 'Standard \u2190 ', fontsize=20, ha='right', color='orange', transform=ax1.transAxes)
    ax1.text(xpos, 1.05, ' \u2192 ZoMBI', fontsize=20, ha='left', color='b', transform=ax1.transAxes)
    xtics = [0, nregular, len(fX_min50) + 1]
    ytics = [lmin, 0., lmax]
    ax1.set_xticks(xtics)
    ax1.set_xticklabels(xtics)
    ax1.set_yticks(ytics)
    ax1.set_yticklabels(np.round(ytics, 2))
    ax1.set_xlabel('Experiments')
    ax1.set_ylabel('Value, $f(X)$')
    ax1.minorticks_on()
    ax1.grid(which='major', linestyle='-', linewidth='0.5', color='grey', alpha=0.8)
    ax1.grid(which='minor', linestyle='-', linewidth='0.5', color='grey', alpha=0.8)
    ax1.axhline(0, lw=1, c='k')

    compute50 = np.percentile(compute, q=50, axis=0)
    compute10 = np.percentile(compute, q=10, axis=0)
    compute90 = np.percentile(compute, q=90, axis=0)
    xs = np.linspace(1, len(compute50) + 1, len(compute50))
    ax2.plot(xs[:nregular], compute50[:nregular], c='orange', lw=2, marker='o', markersize=5, label='Standard')
    ax2.plot(xs[:len(compute50[nregular:]+1)], compute50[nregular:], c='b', lw=2, marker='o', markersize=5, label='ZoMBI')
    ax2.fill_between(xs[:nregular], compute10[:nregular], compute90[:nregular], color = 'orange', alpha = 0.2)
    ax2.fill_between(xs[:len(compute50[nregular:]+1)], compute10[nregular:], compute90[nregular:], color = 'b', alpha=0.2)
    ax2.set_yscale('log')
    ax2.grid(which='major', linestyle='-', linewidth='0.5', color='grey', alpha=0.8)
    ax2.grid(which='minor', linestyle='-', linewidth='0.5', color='grey', alpha=0.8)
    ax2.set_ylabel('Compute Time per\nExperiment [s]')
    ax2.set_xlabel('Experiments')
    ax2.legend(fontsize=12)
    ax1.legend(fontsize=12)
    plt.show();

    plt.figure(figsize=(6,3))
    for n in range(dim * ensemble):
        if n == 0:
            plt.plot(bound_l[:, n], c='k', lw=0.5, label = 'Lower Bound')
            plt.plot(bound_u[:, n], c='r', lw=0.5, label = 'Upper Bound')
        else:
            plt.plot(bound_l[:, n], c='k', lw=0.5)
            plt.plot(bound_u[:, n], c='r', lw=0.5)
    for a in range(activations):
        plt.axvline(a, c='k', lw = 1, ls = ':')
    plt.title('Evolution of Bounds')
    plt.xlabel('ZoMBI Activation')
    plt.ylabel('Bound Value')
    plt.legend(fontsize=12)
    plt.show();