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
        Hint: use self.ekf

        Args:
            state_upd_prev_gauss (MultiVarGaussian): previous update gaussian
            Ts (float): timestep

        Returns:
            state_pred_gauss (MultiVarGaussian): predicted state gaussian
        """

        # TODO replace this with your own code
        state_pred_gauus = solution.pdaf.PDAF.predict_state(
            self, state_upd_prev_gauss, Ts)

        return state_pred_gauus

    def predict_measurement(self, state_pred_gauss: MultiVarGaussian
                            ) -> MultiVarGaussian:
        """Measurement prediction step
        Hint: use self.ekf

        Args:
            state_pred_gauss (MultiVarGaussian): predicted state gaussian

        Returns:
            z_pred_gauss (MultiVarGaussian): predicted measurement gaussian
        """

        # TODO replace this with your own code
        z_pred_gauss = solution.pdaf.PDAF.predict_measurement(
            self, state_pred_gauss)

        return z_pred_gauss

    def gate(self,
             z_pred_gauss: MultiVarGaussian,
             measurements: Sequence[ndarray]) -> ndarray:
        """Gate the incoming measurements. That is remove the measurements 
        that have a mahalanobis distance higher than a certain threshold. 

        Hint: use z_pred_gauss.mahalanobis_distance_sq and self.gate_size_sq

        Args:
            z_pred_gauss (MultiVarGaussian): predicted measurement gaussian 
            measurements (Sequence[ndarray]): sequence of measurements

        Returns:
            gated_measurements (ndarray[:,2]): array of accepted measurements
        """

        # TODO replace this with your own code
        gated_measurements = solution.pdaf.PDAF.gate(
            self, z_pred_gauss, measurements)

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

        # TODO replace this with your own code
        associations_probs = solution.pdaf.PDAF.get_association_prob(
            self, z_pred_gauss, gated_measurements)

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

        # TODO replace this with your own code
        update_gaussians = solution.pdaf.PDAF.get_cond_update_gaussians(
            self, state_pred_gauss, z_pred_gauss, gated_measurements)

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

        # TODO replace this with your own code
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

        # TODO replace this with your own code
        state_pred_gauss, z_pred_gauss, state_upd_gauss = solution.pdaf.PDAF.step_with_info(
            self, state_upd_prev_gauss, measurements, Ts)

        return state_pred_gauss, z_pred_gauss, state_upd_gauss

    def step(self, state_upd_prev_gauss, measurements, Ts):
        _, _, state_upd_gauss = self.step_with_info(state_upd_prev_gauss,
                                                    measurements,
                                                    Ts)
        return state_upd_gauss
