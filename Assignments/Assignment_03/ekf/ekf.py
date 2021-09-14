"""
Notation:
----------
x is generally used for either the state or the mean of a gaussian. It should be clear from context which it is.
P is used about the state covariance
z is a single measurement
Z are multiple measurements so that z = Z[k] at a given time step k
v is the innovation z - h(x)
S is the innovation covariance
"""
from typing import Optional
from dataclasses import dataclass, field
import numpy as np
import scipy.linalg as la

from config import DEBUG
from dynamicmodels import DynamicModel
from measurementmodels import MeasurementModel
from utils.gaussparams import MultiVarGaussian

import solution
# %% The EKF


@dataclass
class EKF:
    dynamic_model: DynamicModel
    sensor_model: MeasurementModel

    def predict(self,
                state_upd_prev_gauss: MultiVarGaussian,
                Ts: float,
                ) -> MultiVarGaussian:
        """Predict the EKF state Ts seconds ahead."""

        # TODO replace this with your own code
        state_pred_gauss = solution.ekf.EKF.predict(
            self, state_upd_prev_gauss, Ts)

        return state_pred_gauss

    def predict_measurement(self,
                            state_pred_gauss: MultiVarGaussian
                            ) -> MultiVarGaussian:
        """Predict measurement pdf from using state pdf and model."""
        x_bar, P = state_pred_gauss
        H = self.sensor_model.H(x_bar)
        R = self.sensor_model.R(x_bar)
        z_bar = np.zeros(2)  # TODO
        S = np.eye(2)  # TODO
        measure_pred_gauss = MultiVarGaussian(z_bar, S)

        # TODO replace this with your own code
        measure_pred_gauss = solution.ekf.EKF.predict_measurement(
            self, state_pred_gauss)

        return measure_pred_gauss

    def update(self,
               z: np.ndarray,
               state_pred_gauss: MultiVarGaussian,
               measurement_gauss: Optional[MultiVarGaussian] = None,
               ) -> MultiVarGaussian:
        """Given the prediction and innovation, 
        find the updated state estimate."""
        x_pred, P = state_pred_gauss
        if measurement_gauss is None:
            measurement_gauss = self.predict_measurement(state_pred_gauss)

        z_bar, S = measurement_gauss
        H = self.sensor_model.H(x_pred)
        x_upd = np.zeros(2)  # TODO
        P_upd = np.eye(2)  # TODO
        state_upd_gauss = MultiVarGaussian(x_upd, P_upd)

        # TODO replace this with your own code
        state_upd_gauss = solution.ekf.EKF.update(
            self, z, state_pred_gauss, measurement_gauss)

        return state_upd_gauss

    def step_with_info(self,
                       state_upd_prev_gauss: MultiVarGaussian,
                       z: np.ndarray,
                       Ts: float,
                       ) -> tuple[MultiVarGaussian,
                                  MultiVarGaussian,
                                  MultiVarGaussian]:
        """
        Predict ekfstate Ts units ahead and then update this prediction with z.

        Returns:
            state_pred_gauss: The state prediction
            measurement_pred_gauss: 
                The measurement prediction after state prediction
            state_upd_gauss: The predicted state updated with measurement
        """

        # TODO replace this with your own code
        state_pred_gauss, measurement_pred_gauss, state_upd_gauss = solution.ekf.EKF.step_with_info(
            self, state_upd_prev_gauss, z, Ts)

        return state_pred_gauss, measurement_pred_gauss, state_upd_gauss

    def step(self,
             state_upd_prev_gauss: MultiVarGaussian,
             z: np.ndarray,
             Ts: float,
             ) -> MultiVarGaussian:

        _, _, state_upd_gauss = self.step_with_info(state_upd_prev_gauss,
                                                    z, Ts)
        return state_upd_gauss
