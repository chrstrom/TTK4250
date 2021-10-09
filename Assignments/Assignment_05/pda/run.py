from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from utils.dataloader import load_data
from utils.interactive_plot import InteractivePlot
from utils.plotting import plot_NEES
from utils.multivargaussian import MultiVarGaussian
from utils.ekf import EKF
from utils.dynamicmodels import WhitenoiseAcceleration2D
from utils.measurementmodels import CartesianPosition2D

import config
from pdaf import PDAF
from tuninghints import tuninghints


def run_pdaf(init_state_gauss, measurement_data, Ts):

    dynamic_model = WhitenoiseAcceleration2D(config.sigma_a)
    sensor_model = CartesianPosition2D(config.sigma_z)
    ekf = EKF(dynamic_model, sensor_model)
    pdaf = PDAF(ekf,
                config.clutter_density,
                config.detection_prob,
                config.gate_percentile)

    state_upd_prev_gauss = init_state_gauss
    filter_data = []
    for measurements in tqdm(measurement_data, "Working",
                             len(measurement_data), None):
        (state_pred_gauss,
         measurement_pred_gauss,
         state_upd_gauss) = pdaf.step_with_info(state_upd_prev_gauss,
                                                measurements, Ts)
        filter_data.append([state_pred_gauss,
                            measurement_pred_gauss,
                            state_upd_gauss])

        state_upd_prev_gauss = state_upd_gauss

    filter_data = list(map(list, zip(*filter_data)))  # transpose list of lists
    return pdaf, filter_data


def main():
    (N_data,
     Ts,
     state_gt_data,
     measurement_data,
     association_gt_data) = load_data()

    tuninghints(measurement_data, association_gt_data)

    init_cov = np.eye(4)*5
    init_mean = np.random.multivariate_normal(state_gt_data[0, :4], np.eye(4))
    init_state_gauss = MultiVarGaussian(init_mean, init_cov)

    pdaf, filter_data = run_pdaf(init_state_gauss, measurement_data, Ts)

    (state_pred_gauss_seq,
     measurement_pred_gauss_seq,
     state_upd_gauss_seq) = filter_data

    pos_upd_gauss_seq = [gauss.marginalize([0, 1])
                         for gauss in state_upd_gauss_seq]

    state: MultiVarGaussian = None
    pos_NEES = [state.marginalize([0, 1]).mahalanobis_distance_sq(gt[:2])
                for state, gt in zip(state_upd_gauss_seq, state_gt_data)]

    pos_error_sq = [np.sum((state.mean[:2] - gt[:2])**2)
                    for state, gt in zip(state_upd_gauss_seq, state_gt_data)]
    pos_RMSE = np.sqrt(sum(pos_error_sq)/len(state_pred_gauss_seq))

    vel_error_sq = [np.sum((state.mean[2:4] - gt[2:4])**2)
                    for state, gt in zip(state_upd_gauss_seq, state_gt_data)]
    vel_RMSE = np.sqrt(sum(vel_error_sq)/len(state_pred_gauss_seq))

    plot_NEES(pos_NEES)
    inter = InteractivePlot(pdaf,
                            state_gt_data[:, :2],
                            pos_upd_gauss_seq,
                            measurement_pred_gauss_seq,
                            measurement_data,
                            association_gt_data,
                            pos_RMSE,
                            vel_RMSE)
    plt.show(block=True)


if __name__ == '__main__':
    main()
