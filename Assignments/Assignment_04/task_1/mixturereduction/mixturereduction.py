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
    mean = None  # TODO

    # internal covariance
    cov_internal = None  # TODO

    # spread of means, aka. external covariance
    # If you vectorize: take care to make the operation order be symetric
    diffs = None  # TODO: optional intermediate
    cov_external = None  # TODO: Hint loop, broadcast or np.einsum

    # total covariance
    cov = None  # TODO

    # TODO replace this with your own code
    mean, cov = solution.mixturereduction.mixture_moments(weights, means, covs)

    return mean, cov
