import numpy as np
from numpy import ndarray
from typing import Sequence, Optional

from datatypes.measurements import GnssMeasurement
from datatypes.eskf_states import NominalState, ErrorStateGauss
from datatypes.multivargaussian import MultiVarGaussStamped

def get_NIS(z_gnss: GnssMeasurement,
            z_gnss_pred_gauss: MultiVarGaussStamped,
            marginal_idxs: Optional[Sequence[int]] = None
            ) -> float:
    """Calculate NIS

    Args:
        z_gnss (GnssMeasurement): gnss measurement
        z_gnss_pred_gauss (MultiVarGaussStamped): predicted gnss measurement
        marginal_idxs (Optional[Sequence[int]]): Sequence of marginal indexes.
            For example used for calculating NIS in only xy direction.  

    Returns:
        NIS (float): NIS value
    """
    z_pos_marg = z_gnss.pos
    z_pred_gauss_marg = z_gnss_pred_gauss

    if marginal_idxs is not None:
        z_pos_marg = z_pos_marg[marginal_idxs]
        z_pred_gauss_marg = z_pred_gauss_marg.marginalize(marginal_idxs)

    nu = z_pos_marg - z_pred_gauss_marg.mean 
    S = z_pred_gauss_marg.cov

    NIS = nu.T@np.linalg.inv(S)@nu
    return NIS


def get_error(x_true: NominalState,
              x_nom: NominalState,
              ) -> 'ndarray[15]':
    """Finds the error (difference) between True state and 
    nominal state. See (Table 10.1).


    Returns:
        error (ndarray[15]): difference between x_true and x_nom. 
    """
    q_inv = x_nom.ori.conjugate()
    error_ori = q_inv @ x_true.ori 
    
    error = np.concatenate([
        x_true.pos - x_nom.pos,
        x_true.vel - x_nom.vel,
        error_ori.as_euler(),
        x_true.accm_bias - x_nom.accm_bias,
        x_true.gyro_bias - x_nom.gyro_bias
    ])

    return error


def get_NEES(error: 'ndarray[15]',
             x_err: ErrorStateGauss,
             marginal_idxs: Optional[Sequence[int]] = None
             ) -> float:
    """Calculate NEES

    Args:
        error (ndarray[15]): errors between x_true and x_nom (from get_error)
        x_err (ErrorStateGauss): estimated error
        marginal_idxs (Optional[Sequence[int]]): Sequence of marginal indexes.
            For example used for calculating NEES for only the position. 

    Returns:
        NEES (float): NEES value
    """
    e = error
    x_err_gauss = x_err
    
    if marginal_idxs is not None:
        e = e[marginal_idxs]
        x_err_gauss = x_err_gauss.marginalize(marginal_idxs)

    P = x_err_gauss.cov

    NEES = e.T@np.linalg.inv(P)@e

    return NEES


def get_time_pairs(unique_data, data):
    """match data from two different time series based on timestamps"""
    gt_dict = dict(([x.ts, x] for x in unique_data))
    pairs = [(gt_dict[x.ts], x) for x in data if x.ts in gt_dict]
    times = [pair[0].ts for pair in pairs]
    return times, pairs
