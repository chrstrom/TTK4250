from dataclasses import dataclass
import numpy as np
from numpy import linalg as nla, ndarray
from functools import cached_property

from config import DEBUG


def isPSD(arr: np.ndarray) -> bool:
    return np.allclose(arr, arr.T) and np.all(np.linalg.eigvals(arr) >= 0)


@dataclass(frozen=True)
class MultiVarGaussian:
    """A class for using Gaussians"""
    mean: ndarray  # shape=(n,)
    cov: ndarray  # shape=(n, n)

    def __post_init__(self):
        if DEBUG:
            assert self.mean.shape * 2 == self.cov.shape
            assert np.all(np.isfinite(self.mean))
            assert np.all(np.isfinite(self.cov))
            assert isPSD(self.cov)

    @cached_property
    def ndim(self) -> int:
        return self.mean.shape[0]

    @cached_property
    def scaling(self) -> float:
        scaling = (2*np.pi)**(-self.ndim/2) * nla.det(self.cov)**(-1/2)
        return scaling

    def mahalanobis_distance_sq(self, x: np.ndarray) -> float:
        """Calculate the mahalanobis distance between self and x.

        This is also known as the quadratic form of the Gaussian.
        See (3.2) in the book.
        """
        # this method could be vectorized for efficient calls
        error = x - self.mean
        mahalanobis_distance = error.T @ nla.solve(self.cov, error)
        return mahalanobis_distance

    def pdf(self, x):
        density = self.scaling*np.exp(-self.mahalanobis_distance_sq(x)/2)
        return density

    def marginalize(self, idxs):
        return MultiVarGaussian(self.mean[idxs], self.cov[idxs][:, idxs])

    def __iter__(self):  # in order to use tuple unpacking
        return iter((self.mean, self.cov))

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, MultiVarGaussian):
            return False
        return np.allclose(self.mean, o.mean) and np.allclose(self.cov, o.cov)
