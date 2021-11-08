import numpy as np
from quaternion import RotationQuaterion
from datatypes.eskf_params import ESKFTuningParams
from datatypes.eskf_states import NominalState, ErrorStateGauss

tuning_params_sim = ESKFTuningParams(
    accm_std=0.01,
    accm_bias_std=0.0005,
    accm_bias_p=10e-12,

    gyro_std=0.0002,
    gyro_bias_std=0.0002,
    gyro_bias_p=10e-12,

    gnss_std_ne=0.25,
    gnss_std_d=0.5)

x_nom_init_sim = NominalState(
    np.array([0., 0., 0.]),  # position
    np.array([0., 0., 0.]),  # velocity
    RotationQuaterion.from_euler([0., 0., 0.]),  # orientation
    np.zeros(3),  # accelerometer bias
    np.zeros(3),  # gyro bias
    ts=0.)

init_std_sim = np.repeat(repeats=3,  # repeat each element 3 times
                         a=[10,  # position
                            10,  # velocity
                            np.deg2rad(np.pi/30),  # angle vector
                            0.001,  # accelerometer bias
                            0.001])  # gyro bias
x_err_init_sim = ErrorStateGauss(np.zeros(15), np.diag(init_std_sim**2), 0.)
