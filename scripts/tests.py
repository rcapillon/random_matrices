import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import src.random_matrices.generators as gen


if __name__ == '__main__':
    N = 10
    dispersion_coeff = 0.2
    n_samples = 100

    mats_GOE = gen.GOE(N, dispersion_coeff, n_samples)
    mats_G_0 = gen.SG_0_plus(N, dispersion_coeff, n_samples)