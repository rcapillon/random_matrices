import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np

import src.random_matrices.generators as gen


def calculate_dispersion_coeff(mats):
    """
    Calculates the dispersion coefficient for given samples of a random matrix
    """
    mean_mat = np.mean(mats, axis=-1)
    norm_mean_mat = np.linalg.norm(mean_mat)
    numerator = 0
    for i in range(mats.shape[-1]):
        numerator += np.linalg.norm(mats[:, :, i] - mean_mat) ** 2
    numerator /= mats.shape[-1]
    dispersion_coeff = np.sqrt(numerator / norm_mean_mat ** 2)

    return dispersion_coeff


if __name__ == '__main__':
    N = 5
    delta = 0.3
    n_samples = 1000

    mats_GOE = gen.GOE(N, delta, n_samples)
    mean_mat_GOE = np.mean(mats_GOE, axis=-1)
    delta_calc_GOE = calculate_dispersion_coeff(mats_GOE)
    print('GOE: prescribed delta')
    print(delta)
    print('GOE: estimated delta')
    print(delta_calc_GOE)
    # print('GOE: mean matrix')
    # print(mean_mat_GOE)
    print('\n')

    mats_G_0 = gen.SG_0_plus(N, delta, n_samples)
    mean_mat_G_0 = np.mean(mats_G_0, axis=-1)
    delta_calc_G_0 = calculate_dispersion_coeff(mats_G_0)
    print('SG_O+: prescribed delta')
    print(delta)
    print('SG_O+: estimated delta')
    print(delta_calc_G_0)
    # print('SG_O+: mean matrix')
    # print(mean_mat_G_0)
    print('\n')

    mats_G_eps = gen.SG_eps_plus(N, delta, n_samples, eps=1e-3)
    mean_mat_G_eps = np.mean(mats_G_eps, axis=-1)
    delta_calc_G_eps = calculate_dispersion_coeff(mats_G_eps)
    print('SG_eps+: prescribed delta')
    print(delta)
    print('SG_eps+: estimated delta')
    print(delta_calc_G_eps)
    # print('SG_eps+: mean matrix')
    # print(mean_mat_G_eps)
    print('\n')

    mean_mat = np.diag([2] * (N - 1), k=-1) + np.diag([10] * N, k=0) + np.diag([2] * (N - 1), k=1)
    mats_SE_0_plus = gen.SE_0_plus(delta, mean_mat, n_samples)
    mean_mat_SE_0_plus = np.mean(mats_SE_0_plus, axis=-1)
    delta_calc_SE_0_plus = calculate_dispersion_coeff(mats_SE_0_plus)
    print('SE_0_+: prescribed delta')
    print(delta)
    print('SE_0_+: estimated delta')
    print(delta_calc_SE_0_plus)
    # print('SE_0_+: mean matrix')
    # print(mean_mat_SE_0_plus)
    print('\n')

    mean_mat = np.array([[1, -1], [-1, 1]])
    mats_SE_plus0 = gen.SE_plus0(delta, mean_mat, n_samples, eps=1e-3, tol=1e-9)
    mean_mat_SE_plus0 = np.mean(mats_SE_plus0, axis=-1)
    delta_calc_SE_plus0 = calculate_dispersion_coeff(mats_SE_plus0)
    print('SE_+0: prescribed delta')
    print(delta)
    print('SE_+0: estimated delta')
    print(delta_calc_SE_plus0)
    # print('SE_+0: mean matrix')
    # print(mean_mat_SE_plus0)
    print('\n')

    mean_mat = np.array([[1, 2], [3, 4], [5, 6]])
    mats_SE_rect = gen.SE_rect(delta, mean_mat, n_samples, eps=1e-3)
    mean_mat_SE_rect = np.mean(mats_SE_rect, axis=-1)
    delta_calc_SE_rect = calculate_dispersion_coeff(mats_SE_rect)
    print('SE_rect: prescribed delta')
    print(delta)
    print('SE_rect: estimated delta')
    print(delta_calc_SE_rect)
    # print('SE_rect: mean matrix')
    # print(mean_mat_SE_rect)
    print('\n')
