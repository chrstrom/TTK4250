import numpy as np
from numpy import ndarray
from scipy.stats import chi2
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

from gaussmix import GaussianMuxture
from utils.multivargaussian import MultiVarGaussian
from utils.ekf import EKF

import solution


@dataclass
class PDAF:

    ekf: EKF
    clutter_density: float
    detection_prob: float
    gate_percentile: float
    gate_size_sq: float = field(init=False)

    def __post_init__(self):
        self.gate_size_sq = chi2.ppf(self.gate_percentile,
                                     self.ekf.sensor_model.ndim)

    def predict_state(self, state_upd_prev_gauss: MultiVarGaussian, Ts: float
                      ) -> MultiVarGaussian:
        """Prediction step

        Args:
            state_upd_prev_gauss (MultiVarGaussian): previous update gaussian
            Ts (float): timestep

        Returns:
            state_pred_gauss (MultiVarGaussian): predicted state gaussian
        """
        state_pred_gauss = self.ekf.predict_state(state_upd_prev_gauss, Ts)
        return state_pred_gauss

    def predict_measurement(self, state_pred_gauss: MultiVarGaussian
                            ) -> MultiVarGaussian:
        """Measurement prediction step

        Args:
            state_pred_gauss (MultiVarGaussian): predicted state gaussian

        Returns:
            z_pred_gauss (MultiVarGaussian): predicted measurement gaussian
        """
        z_pred_gauss = self.ekf.predict_measurement(state_pred_gauss)
        return z_pred_gauss

    def gate(self,
             z_pred_gauss: MultiVarGaussian,
             measurements: Sequence[ndarray]) -> ndarray:
        """Gate the incoming measurements. That is remove the measurements 
        that have a mahalanobis distance higher than a certain threshold. 

        Args:
            z_pred_gauss (MultiVarGaussian): predicted measurement gaussian 
            measurements (Sequence[ndarray]): sequence of measurements

        Returns:
            gated_measurements (ndarray[:,2]): array of accepted measurements
        """
        gated_measurements = [
            m for m in measurements
            if z_pred_gauss.mahalanobis_distance_sq(m) < self.gate_size_sq
        ]

        return gated_measurements

    def get_association_prob(self, z_pred_gauss: MultiVarGaussian,
                             gated_measurements: ndarray
                             ) -> ndarray:
        """Finds the association probabilities.

        associations_probs[0]: prob that no association is correct
        associations_probs[1]: prob that gated_measurements[0] is correct
        associations_probs[2]: prob that gated_measurements[1] is correct
        ...

        the sum of associations_probs should be 1

        Args:
            z_pred_gauss (MultiVarGaussian): predicted measurement gaussian 
            gated_measurements (ndarray[:,2]): array of accepted measurements

        Returns:
            associations_probs (ndarray[:]): the association probabilities
        """

        m_k = len(gated_measurements)
        P_D = self.detection_prob
    
        # Implementing Corollary 7.3.2
        associations_probs = []
        associations_probs.append(m_k * (1 - P_D) * self.clutter_density)   # a_k = 0
        for i in range(m_k):
            associations_probs.append(P_D*z_pred_gauss.pdf(gated_measurements[i]))    # a_k > 0
        
        associations_probs = np.array(associations_probs)
        if associations_probs.sum() != 0:
            associations_probs /= associations_probs.sum()
        else:
            associations_probs += 1/associations_probs.size


        return associations_probs

    def get_cond_update_gaussians(self, state_pred_gauss: MultiVarGaussian,
                                  z_pred_gauss: MultiVarGaussian,
                                  gated_measurements: ndarray
                                  ) -> Sequence[MultiVarGaussian]:
        """Get the conditional updated state gaussians 
        for every association hypothesis

        update_gaussians[0]: update given that no measurement is correct
        update_gaussians[1]: update given that gated_measurements[0] is correct
        update_gaussians[2]: update given that gated_measurements[1] is correct
        ...

        Args:
            state_pred_gauss (MultiVarGaussian): predicted state gaussian
            z_pred_gauss (MultiVarGaussian): predicted measurement gaussian
            gated_measurements (ndarray[:,2]): array of accepted measurements

        Returns:
            Sequence[MultiVarGaussian]: The sequence of conditional updates
        """
        x_pred, P_pred = state_pred_gauss
        z_pred, S_pred = z_pred_gauss
        H = self.ekf.sensor_model.jac(x_pred)
        W = P_pred@H.T@np.linalg.inv(S_pred)

        # Implementing 7.20 and 7.21
        update_gaussians = []
        update_gaussians.append(MultiVarGaussian(x_pred, P_pred)) #a_k = 0
        for z_k in gated_measurements:
            mean = x_pred + W@(z_k - z_pred)
            cov = (np.eye(4) - W@H)@P_pred
            update_gaussians.append(MultiVarGaussian(mean, cov))

        return update_gaussians

    def update(self, state_pred_gauss: MultiVarGaussian,
               z_pred_gauss: MultiVarGaussian,
               measurements: Sequence[ndarray]):
        """Perform the update step of the PDA filter

        Args:
            state_pred_gauss (MultiVarGaussian): predicted state gaussian
            z_pred_gauss (MultiVarGaussian): predicted measurement gaussian
            measurements (Sequence[ndarray]): sequence of measurements

        Returns:
            state_upd_gauss (MultiVarGaussian): updated state gaussian
        """

        gated_measurements = self.gate(z_pred_gauss, measurements)
        beta = self.get_association_prob(z_pred_gauss, gated_measurements)
        conditional_gaussians = self.get_cond_update_gaussians(state_pred_gauss, z_pred_gauss, gated_measurements)

        # Not sure why this one isn't working
        #state_upd_gauss = GaussianMuxture(beta, conditional_gaussians).reduce()

        state_upd_gauss = solution.pdaf.PDAF.update(
            self, state_pred_gauss, z_pred_gauss, measurements)
        return state_upd_gauss

    def step_with_info(self,
                       state_upd_prev_gauss: MultiVarGaussian,
                       measurements: Sequence[ndarray],
                       Ts: float
                       ) -> Tuple[MultiVarGaussian,
                                  MultiVarGaussian,
                                  MultiVarGaussian]:
        """Perform a full step and return usefull info

        Hint: you should not need to write any new code here, 
        just use the methods you have implemented

        Args:
            state_upd_prev_gauss (MultiVarGaussian): previous updated gaussian
            measurements (Sequence[ndarray]): sequence of measurements
            Ts (float): timestep

        Returns:
            state_pred_gauss (MultiVarGaussian): predicted state gaussian
            z_pred_gauss (MultiVarGaussian): predicted measurement gaussian
            state_upd_gauss (MultiVarGaussian): updated state gaussian
        """
        state_pred_gauss = self.predict_state(state_upd_prev_gauss, Ts)
        z_pred_gauss = self.predict_measurement(state_pred_gauss)
        state_upd_gauss = self.update(state_pred_gauss, z_pred_gauss, measurements)

        return state_pred_gauss, z_pred_gauss, state_upd_gauss

    def step(self, state_upd_prev_gauss, measurements, Ts):
        _, _, state_upd_gauss = self.step_with_info(state_upd_prev_gauss,
                                                    measurements,
                                                    Ts)
        return state_upd_gauss
