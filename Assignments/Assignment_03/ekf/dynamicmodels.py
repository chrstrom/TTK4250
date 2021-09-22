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

        #x_kp1 = solution.dynamicmodels.WhitenoiseAcceleration2D.f(self, x, Ts)
        x_kp1 = self.F(x, Ts) @ x
        return x_kp1

    def F(self, x: ndarray, Ts: float,) -> ndarray:
        """Calculate the transition function jacobian for Ts time units at x.
        See DynamicModel for variable documentation"""
        #F = solution.dynamicmodels.WhitenoiseAcceleration2D.F(self, x, Ts)
        F = np.array([[1, 0, Ts,  0],
                      [0, 1,  0, Ts],
                      [0, 0,  1,  0],
                      [0, 0,  0,  1]])

        return F

    def Q(self, x: ndarray, Ts: float,) -> ndarray:
        """Calculate the Ts time units transition Covariance.
        See(4.64) in the book.
        See DynamicModel for variable documentation"""
        T3 = Ts**3 / 3.0
        T2 = Ts**2 / 2.0

        Q = np.array([[T3,  0, T2,  0],
                      [ 0, T3,  0, T2],
                      [T2,  0, Ts,  0],
                      [ 0, T2,  0, Ts]]) * self.sigma_a**2
        return Q
