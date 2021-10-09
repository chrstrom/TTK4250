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
from dataclasses import dataclass
import numpy as np
import scipy.linalg as la

from utils.dynamicmodels import DynamicModel
from utils.measurementmodels import MeasurementModel
from utils.multivargaussian import MultiVarGaussian


@dataclass
class EKF:
    dynamic_model: DynamicModel
    sensor_model: MeasurementModel

    def predict_state(self,
                      state_upd_prev_gauss: MultiVarGaussian,
                      Ts: float,
                      ) -> MultiVarGaussian:
        """Predict the EKF state Ts seconds ahead."""
        x_upd_prev, P = state_upd_prev_gauss

        F = self.dynamic_model.jac(x_upd_prev, Ts)
        Q = self.dynamic_model.Q(x_upd_prev, Ts)

        x_pred = self.dynamic_model.f(x_upd_prev, Ts)
        P_pred = F @ P @ F.T + Q

        state_pred_gauss = MultiVarGaussian(x_pred, P_pred)

        return state_pred_gauss

    def predict_measurement(self,
                            state_pred_gauss: MultiVarGaussian
                            ) -> MultiVarGaussian:
        """Predict measurement pdf from using state pdf and model."""
        if False:
            x_bar, P = state_pred_gauss
            H = self.sensor_model.H(x_bar)
            R = self.sensor_model.R(x_bar)
            z_bar = np.zeros(2)  # TODO
            S = np.eye(2)  # TODO
            measure_pred_gauss = MultiVarGaussian(z_bar, S)

        x_pred, P = state_pred_gauss

        # Calculate mean of measurement
        z_pred = self.sensor_model.h(x_pred)

        # Calculate the measurement (innovation) covariance
        # for ekfstate at z in sensorstate.
        H = self.sensor_model.jac(x_pred)
        R = self.sensor_model.R(x_pred)
        S = H @ P @ H.T + R

        # Create Gaussian
        measure_pred_gauss = MultiVarGaussian(z_pred, S)

        return measure_pred_gauss

    def update(self,
               state_pred_gauss: MultiVarGaussian,
               measurement: np.ndarray,
               measurement_pred_gauss: Optional[MultiVarGaussian] = None,
               ) -> MultiVarGaussian:
        """Given the prediction and innovation, 
        find the updated state estimate."""

        x_pred, P = state_pred_gauss

        if measurement_pred_gauss is None:
            measurement_pred_gauss = self.predict_measurement(state_pred_gauss)

        z_bar, S = measurement_pred_gauss
        innovation = measurement - z_bar

        H = self.sensor_model.jac(x_pred)

        # Kalman gain
        W = P @ la.solve(S, H).T

        # mean update
        x_upd = x_pred + W @ innovation

        # covariance update
        # P_upd = P - W @ H @ P  # simple standard form

        # It might be better to use the more numerically stable Joseph form
        I = np.eye(*P.shape)
        P_upd = ((I - W @ H) @ P @ (I - W @ H).T
                 + W @ self.sensor_model.R(state_pred_gauss) @ W.T)

        state_upd_gauss = MultiVarGaussian(x_upd, P_upd)

        return state_upd_gauss

    def step_with_info(self,
                       state_upd_prev_gauss: MultiVarGaussian,
                       measurement: np.ndarray,
                       Ts: float,
                       ) -> tuple[MultiVarGaussian,
                                  MultiVarGaussian,
                                  MultiVarGaussian]:
        """Predict ekfstate Ts units ahead and then 
        update this prediction with z.

        Returns:
            state_pred_gauss: The state prediction
            measurement_pred_gauss: 
                The measurement prediction after state prediction
            state_upd_gauss: The predicted state updated with measurement
        """

        state_pred_gauss = self.predict_state(state_upd_prev_gauss, Ts)
        measurement_pred_gauss = self.predict_measurement(state_pred_gauss)
        state_upd_gauss = self.update(state_pred_gauss, measurement,
                                      measurement_pred_gauss)

        return state_pred_gauss, measurement_pred_gauss, state_upd_gauss

    def step(self,
             state_upd_prev_gauss: MultiVarGaussian,
             measurement: np.ndarray,
             Ts: float,
             ) -> MultiVarGaussian:

        _, _, state_upd_gauss = self.step_with_info(state_upd_prev_gauss,
                                                    measurement, Ts)
        return state_upd_gauss
