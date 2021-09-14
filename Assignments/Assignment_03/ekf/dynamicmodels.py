#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic models to be used with eg. EKF.

@author: Lars-Christian Tokle, lars-christian.n.tokle@ntnu.no
"""
# %%
from dataclasses import dataclass

import numpy as np
from numpy import ndarray
import solution


@dataclass
class DynamicModel:
    """
    Parent class for dynamic models.

    A model includes the discrete prediction equation f, its Jacobian F, and
    the process noise covariance Q.
    """

    def f(self, x: ndarray, Ts: float, **kwargs) -> ndarray:
        """Calculate the zero noise Ts time units transition from x.

          Args:
              x (ndarray): state
              Ts (float): time step

          Returns:
              x_kp1 (ndarray): x_k+1, the next state
          """
        raise NotImplementedError

    def F(self, x: ndarray, Ts: float, **kwargs) -> ndarray:
        """Calculate the transition function jacobian for Ts time units at x.
        Args:
            x (ndarray): state
            Ts (float): time step

        Returns:
            F (ndarray): Discrete ransition function jacobian,
                         for linear systems: x_k+1 = F @ x_k
        """
        raise NotImplementedError

    def Q(self, x: ndarray, Ts: float, **kwargs) -> ndarray:
        """Calculate the Ts time units transition Covariance.
        Args:
            x (ndarray): state
            Ts (float): time step

        Returns:
            Q (ndarray): covariance matrix
        """
        raise NotImplementedError


@dataclass
class WhitenoiseAcceleration2D(DynamicModel):
    """
    A white noise acceleration model, also known as constan velocity.
    States are position and speed.
    """

    # noise standard deviation
    sigma_a: float

    def f(self, x: ndarray, Ts: float,) -> ndarray:
        """Calculate the zero noise Ts time units transition from x.
        See DynamicModel for variable documentation
        """

        # TODO replace this with your own code
        x_kp1 = solution.dynamicmodels.WhitenoiseAcceleration2D.f(self, x, Ts)

        return x_kp1

    def F(self, x: ndarray, Ts: float,) -> ndarray:
        """Calculate the transition function jacobian for Ts time units at x.
        See DynamicModel for variable documentation"""

        # TODO replace this with your own code
        F = solution.dynamicmodels.WhitenoiseAcceleration2D.F(self, x, Ts)

        return F

    def Q(self, x: ndarray, Ts: float,) -> ndarray:
        """Calculate the Ts time units transition Covariance.
        See(4.64) in the book.
        See DynamicModel for variable documentation"""

        # TODO replace this with your own code
        Q = solution.dynamicmodels.WhitenoiseAcceleration2D.Q(self, x, Ts)

        return Q
