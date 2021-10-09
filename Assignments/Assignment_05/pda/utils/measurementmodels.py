# %% Imports
from typing import Any, Dict
from dataclasses import dataclass
import numpy as np
from numpy import ndarray
# %% Measurement models interface declaration


@dataclass
class MeasurementModel:
    def h(self, x: ndarray, **kwargs) -> ndarray:
        """Calculate the noise free measurement location at x in sensor_state.
        Args:
            x (ndarray): state
        """
        raise NotImplementedError

    def jac(self, x: ndarray, **kwargs) -> ndarray:
        """Calculate the measurement Jacobian matrix at x in sensor_state.
        Args:
            x (ndarray): state
        """
        raise NotImplementedError

    def R(self, x: ndarray, **kwargs) -> ndarray:
        """Calculate the measurement covariance matrix at x in sensor_state.
        Args:
            x (ndarray): state
        """
        raise NotImplementedError


@dataclass
class CartesianPosition2D(MeasurementModel):
    sigma_z: float
    ndim: int = 2

    def h(self, x: ndarray) -> ndarray:
        """Calculate the noise free measurement location at x in sensor_state.
        """
        x_h = x[:2]
        return x_h

    def jac(self, x: ndarray) -> ndarray:
        """Calculate the measurement Jacobian matrix at x in sensor_state."""
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        return H

    def R(self, x: ndarray) -> ndarray:
        """Calculate the measurement covariance matrix at x in sensor_state."""
        R = self.sigma_z**2 * np.eye(2)
        return R
