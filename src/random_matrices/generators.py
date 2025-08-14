import numpy as np
from scipy.stats import norm


def GOE(N, dispersion_coeff, n_samples):
    """
    Generator for random matrices belonging to the Gaussian Orthogonal Ensemble
    """
    mats_I = np.tile(np.eye(N)[:, :, np.newaxis], (1, 1, n_samples))
    mats_GOE_delta = np.zeros((N, N, n_samples))
    for j in range(N):
        for k in range(N):
            if j == k:
                sigma = np.sqrt(2 * dispersion_coeff ** 2 / (N + 1))
            else:
                sigma = np.sqrt(dispersion_coeff ** 2 / (N + 1))
            component_samples = sigma * norm.rvs(n_samples)
            mats_GOE_delta[j, k, :] = component_samples
            mats_GOE_delta[k, j, :] = component_samples
    mats_GOE = mats_I + mats_GOE_delta

    return mats_GOE


def SG_0_plus(N, dispersion_coeff, n_samples):
    """
    Generator for random matrices belonging to the SG_0_+ ensemble
    (positive-definite matrices with identity matrix as mean value)
    """
    if dispersion_coeff <= 0 or dispersion_coeff >= np.sqrt((N + 1) / (N + 5)):
        raise ValueError('Dispersion coefficient should be greater than 0 '
                         'and smaller than sqrt((N + 1) / (N + 5))')

    mats_L = np.zeros((N, N, n_samples))
    for j in range(N):
        for k in range(j + 1):
            if j == k:
                pass
            else:
                sigma = dispersion_coeff / np.sqrt(N + 1)
                component_samples = sigma * norm.rvs(n_samples)
                mats_L[k, j, :] = component_samples
    mats_G0 = np.tensordot(np.transpose(mats_L, axes=(0, 1)), mats_L, axes=(1, 0))

    return mats_G0

