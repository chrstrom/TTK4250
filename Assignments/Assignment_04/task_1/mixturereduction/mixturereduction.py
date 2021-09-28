import numpy as np
from numpy import ndarray

import solution


def mixture_moments(weights: ndarray,
                    means: ndarray,
                    covs: ndarray,
                    ) -> tuple[ndarray, ndarray]:
    """Calculate the first two moments of a Gaussian mixture.

    Args:
        weights: shape = (N,)
        means: shape = (N, n)
        covs: shape = (N, n, n)

    Returns:
        mean: shape = (n,)
        cov: shape = (n, n)
    """
    
    N = means.shape[0]
    n = means.shape[1]

    mean = 0
    for i in range(N):
        mean += weights[i]*means[i]

    cov = np.empty([n, n])
    for i in range(N):
        diff = means[i] - mean
        cov += weights[i]*covs[i] + weights[i]*(diff@diff.T)

    return mean, cov
