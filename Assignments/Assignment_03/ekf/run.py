# %% Imports
from typing import Collection
from scipy import stats
from matplotlib import pyplot as plt
import numpy as np
import numpy.linalg as nla
from numpy import ndarray
from tqdm import tqdm

from utils.dataloader import load_data
from utils.plotting import (plot_trajectory_with_measurements,
                            plot_ekf_trajectory,
                            plot_NIS_NEES_data)

from ekf import EKF
from utils.gaussparams import MultiVarGaussian
from measurementmodels import CartesianPosition2D
from dynamicmodels import WhitenoiseAcceleration2D
from analysis import get_ANIS, get_ANEES


def run_ekf(sigma_a: float, sigma_z: float,
            z_data: Collection[ndarray], Ts: float, N_data: int):
    """This function will estimate the initial state and covariance from
    the measurements and iterate the kalman filter through the data.

    Args:
        sigma_a (float): std of acceleration
        sigma_z (float): std of measurements
        Ts (float): the time step between each measurement
        N_data (int): the number of measurements
    Returns:
        state_pred_gauss_data (list[MultiVarGaussian]):
            list of all state predictions
        measurement_gauss_data (list[MultiVarGaussian]):
            list of all measurement pdfs
        state_upd_gauss_data (list[MultiVarGaussian]):
            list of all updated states
    """
    # create the model and estimator object
    dynmod = WhitenoiseAcceleration2D(sigma_a)
    measmod = CartesianPosition2D(sigma_z)
    ekf_filter = EKF(dynmod, measmod)

    # Optimal init for model
    mean = np.array([*z_data[1], *(z_data[1] - z_data[0]) / Ts])
    cov11 = sigma_z ** 2 * np.eye(2)
    cov12 = sigma_z ** 2 * np.eye(2) / Ts
    cov22 = (2 * sigma_z ** 2 / Ts ** 2 + sigma_a ** 2 * Ts / 3) * np.eye(2)
    cov = np.block([[cov11, cov12], [cov12.T, cov22]])
    init_ekfstate = MultiVarGaussian(mean, cov)

    # estimate
    x_upd_gauss = init_ekfstate
    x_pred_gauss_data = []
    z_pred_gauss_data = []
    x_upd_gauss_data = []
    NIS_data = []
    for z_k in z_data[2:]:

        (x_pred_gauss,
         z_pred_gauss,
         x_upd_gauss) = ekf_filter.step_with_info(x_upd_gauss, z_k, Ts)

        x_pred_gauss_data.append(x_pred_gauss)
        z_pred_gauss_data.append(z_pred_gauss)
        x_upd_gauss_data.append(x_upd_gauss)

    return x_pred_gauss_data, z_pred_gauss_data, x_upd_gauss_data


def show_ekf_output(sigma_a: float, sigma_z: float,
                    x_gt_data: Collection[ndarray],
                    z_data: Collection[ndarray], Ts: float, N_data: int):
    """Run the calman filter, find RMSE and show the trajectory"""

    (x_pred_gauss,
     z_pred_gauss,
     x_upd_gauss) = run_ekf(sigma_a, sigma_z, z_data, Ts, N_data)

    x_hat_data = np.array([upd.mean[:2] for upd in x_upd_gauss])

    diff_pred_data = np.array([pred.mean - x_gt[:4] for pred, x_gt
                               in zip(x_pred_gauss, x_gt_data)])

    diff_upd_data = np.array([upd.mean - x_gt[:4]for upd, x_gt
                              in zip(x_upd_gauss, x_gt_data)])

    RMSE_pred = np.sqrt(
        np.mean(np.sum(diff_pred_data.reshape(-1, 2, 2)**2, axis=-1), axis=0))
    RMSE_upd = np.sqrt(
        np.mean(np.sum(diff_upd_data.reshape(-1, 2, 2)**2, axis=-1), axis=0))

    plot_ekf_trajectory(x_gt_data, x_hat_data,
                        RMSE_pred, RMSE_upd, sigma_a, sigma_z)
# %% Task 5 b and c


def try_multiple_alphas(x_gt_data: Collection[ndarray],
                        z_data: Collection[ndarray],
                        Ts: float, N_data: int):
    """Run the Kalman filter with multiple different sigma values,
    the result from each run is used to create a mesh plot of the NIS and NEES
    values for the different configurations"""
    # % parameters for the parameter grid
    n_vals = 20
    sigma_a_low = 0.5
    sigma_a_high = 10
    sigma_z_low = 0.3
    sigma_z_high = 12

    # % set the grid on logscale(not mandatory)
    sigma_a_list = np.geomspace(sigma_a_low, sigma_a_high, n_vals)
    sigma_z_list = np.geomspace(sigma_z_low, sigma_z_high, n_vals)

    ANIS_data = np.empty((n_vals, n_vals))
    ANEES_pred_data = np.empty((n_vals, n_vals))
    ANEES_upd_data = np.empty((n_vals, n_vals))

    # tqdm is used to show progress bars
    for i, sigma_a in tqdm(enumerate(sigma_a_list), "sigma_a", n_vals, None):
        for j, sigma_z in tqdm(enumerate(sigma_z_list),
                               "sigma_z", n_vals, None):

            (x_pred_gauss_data,
             z_pred_gauss_data,
             x_upd_gauss_data) = run_ekf(sigma_a, sigma_z, z_data,
                                         Ts, N_data)

            # dont use the first 2 values of x_gt_data or a_data
            # as they are used for initialzation

            ANIS_data[i, j] = get_ANIS(z_pred_gauss_data, z_data[2:])

            ANEES_pred_data[i, j] = get_ANEES(x_pred_gauss_data,
                                              x_gt_data[2:, :4])

            ANEES_upd_data[i, j] = get_ANEES(x_upd_gauss_data,
                                             x_gt_data[2:, :4])

    confprob = 0.9
    CINIS = np.array(stats.chi2.interval(confprob, 2 * N_data)) / N_data
    CINEES = np.array(stats.chi2.interval(confprob, 4 * N_data)) / N_data
    plot_NIS_NEES_data(sigma_a_low, sigma_a_high, sigma_a_list,
                       sigma_z_low, sigma_z_high, sigma_z_list,
                       ANIS_data, CINIS,
                       ANEES_pred_data, ANEES_upd_data, CINEES)


def main():
    usePregen = True  # choose between own generated data and pregenerated
    x_gt_data, z_data, Ts, N_data = load_data(usePregen)
    plot_trajectory_with_measurements(x_gt_data, z_data)

    # %% a: tune by hand and comment

    # set parameters
    sigma_a = 2.6
    sigma_z = 3.1

    show_ekf_output(sigma_a, sigma_z, x_gt_data, z_data, Ts, N_data)

    if input("Try multiple alpha combos? (y/n): ") == 'y':
        try_multiple_alphas(x_gt_data, z_data, Ts, N_data)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
