import numpy as np
from numpy import ndarray
import scipy.linalg as la
import solution
from utils.gaussparams import MultiVarGaussian
from config import DEBUG
from typing import Sequence


def get_NIS(z_pred_gauss: MultiVarGaussian, z: ndarray):
    """Calculate the normalized innovation squared (NIS), this can be seen as 
    the normalized measurement prediction error squared. 
    See (4.66 in the book). 
    Tip: use the mahalanobis_distance method of z_pred_gauss, (3.2) in the book

    Args:
        z_pred_gauss (MultiVarGaussian): predigted measurement gaussian
        z (ndarray): measurement

    Returns:
        NIS (float): normalized innovation squared
    """

    # TODO replace this with your own code
    NIS = solution.analysis.get_NIS(z_pred_gauss, z)

    return NIS


def get_NEES(x_gauss: MultiVarGaussian, x_gt: ndarray):
    """Calculate the normalized estimation error squared (NEES)
    See (4.65 in the book). 
    Tip: use the mahalanobis_distance method of x_gauss, (3.2) in the book

    Args:
        x_gauss (MultiVarGaussian): state estimate gaussian
        x_gt (ndarray): true state

    Returns:
        NEES (float): normalized estimation error squared
    """

    # TODO replace this with your own code
    NEES = solution.analysis.get_NEES(x_gauss, x_gt)

    return NEES


def get_ANIS(z_pred_gauss_data: Sequence[MultiVarGaussian],
             z_data: Sequence[ndarray]):
    """Calculate the average normalized innovation squared (ANIS)
    Tip: use get_NIS

    Args:
        z_pred_gauss_data (Sequence[MultiVarGaussian]): Sequence (List) of 
            predicted measurement gaussians
        z_data (Sequence[ndarray]): Sequence (List) of true measurements

    Returns:
        ANIS (float): average normalized innovation squared
    """

    # TODO replace this with your own code
    ANIS = solution.analysis.get_ANIS(z_pred_gauss_data, z_data)

    return ANIS


def get_ANEES(x_upd_gauss_data: Sequence[MultiVarGaussian],
              x_gt_data: Sequence[ndarray]):
    """Calculate the average normalized estimation error squared (ANEES)
    Tip: use get_NEES

    Args:
        x_upd_gauss_data (Sequence[MultiVarGaussian]): Sequence (List) of 
            state estimate gaussians
        x_gt_data (Sequence[ndarray]): Sequence (List) of true states

    Returns:
        ANEES (float): average normalized estimation error squared
    """

    # TODO replace this with your own code
    ANEES = solution.analysis.get_ANEES(x_upd_gauss_data, x_gt_data)

    return ANEES
