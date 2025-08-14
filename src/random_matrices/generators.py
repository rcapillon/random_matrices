import numpy as np
from scipy.stats import norm, gamma


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
            component_samples = sigma * norm.rvs(size=n_samples)
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

    sigma = dispersion_coeff / np.sqrt(N + 1)
    mats_L = np.zeros((N, N, n_samples))
    for j in range(N):
        for k in range(j + 1):
            if j == k:
                alpha = (N + 1) / (2 * dispersion_coeff ** 2) + (1 - k) / 2
                component_samples = sigma * np.sqrt(2 * gamma.rvs(alpha, size=n_samples))
                mats_L[j, j, :] = component_samples
            else:
                component_samples = sigma * norm.rvs(size=n_samples)
                mats_L[k, j, :] = component_samples
    mats_G_0 = np.zeros((N, N, n_samples))
    for i in range(n_samples):
        mats_G_0[:, :, i] = np.dot(np.transpose(mats_L[:, :, i]), mats_L[:, :, i])

    return mats_G_0


def SG_eps_plus(N, dispersion_coeff, n_samples, eps=1e-3):
    """
    Generator for random matrices belonging to the SG_eps_+ ensemble
    (positive-definite matrices with identity matrix as mean value and with a positive lower bound)
    """
    delta_G_0 = (1 + eps) * dispersion_coeff
    mats_G_0 = SG_0_plus(N, delta_G_0, n_samples)
    mats_I = np.tile(np.eye(N)[:, :, np.newaxis], (1, 1, n_samples))
    mats_G_eps = (mats_G_0 + eps * mats_I) / (1 + eps)

    return mats_G_eps


def SE_0_plus(dispersion_coeff, mean_mat, n_samples):
    """
    Generator for random matrices belonging to the SE_0_plus ensemble
    (positive-definite matrices with given mean value)
    """
    N = mean_mat.shape[0]
    L_upper = np.linalg.cholesky(mean_mat, upper=True)
    delta_G_0 = (dispersion_coeff *
                 np.sqrt((N + 1) / (1 + (np.linalg.trace(mean_mat) ** 2) / np.linalg.norm(mean_mat) ** 2)))
    mats_G_0 = SG_0_plus(N, delta_G_0, n_samples)
    mats_SE_0_plus = np.zeros((N, N, n_samples))
    for i in range(n_samples):
        mats_SE_0_plus[:, :, i] = np.dot(L_upper.T, np.dot(mats_G_0[:, :, i], L_upper))

    return mats_SE_0_plus


def SE_plus0(dispersion_coeff, mean_mat, n_samples, eps=1e-3, tol=1e-9):
    """
    Generator for random matrices belonging to the SE_plus0 ensemble
    (positive semidefinite matrices with given mean value)
    """
    eigvals, eigvects = np.linalg.eig(mean_mat)
    eigvects = eigvects[:, eigvals >= tol]
    eigvals = eigvals[eigvals >= tol]
    mat_R = np.dot(eigvects, np.diag(np.sqrt(eigvals)))
    n = eigvals.size
    mats_G_eps = SG_eps_plus(n, dispersion_coeff, n_samples, eps)
    N = mean_mat.shape[0]
    mats_SE_plus0 = np.zeros((N, N, n_samples))
    for i in range(n_samples):
        mats_SE_plus0[:, :, i] = np.dot(mat_R, np.dot(mats_G_eps[:, :, i], mat_R.T))

    return mats_SE_plus0


def SE_rect(dispersion_coeff, mean_mat, n_samples, eps=1e-3):
    """
    Generator for random matrices belonging to the SE_rect ensemble
    (rectangular matrices with given mean value)
    """
    U, S, Vh = np.linalg.svd(mean_mat, full_matrices=False)
    mat_U = np.dot(U, Vh)
    mat_L = np.dot(np.diag(np.sqrt(S)), Vh)
    M = mat_U.shape[0]
    N = mat_L.shape[0]
    mats_G_eps = SG_eps_plus(N, dispersion_coeff, n_samples, eps)
    mats_SE_rect = np.zeros((M, N, n_samples))
    for i in range(n_samples):
        mats_SE_rect[:, :, i] = np.dot(mat_U, np.dot(mat_L.T, np.dot(mats_G_eps[:, :, i], mat_L)))

    return mats_SE_rect
