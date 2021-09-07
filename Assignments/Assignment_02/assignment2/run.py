import numpy as np
import matplotlib.pyplot as plt

from utils.plotting import show_task_2f_and_2g, show_task_2h
from utils.interactive_covariance import InteractiveCovariance
from task2 import (condition_mean, condition_cov,
                   get_task_2f, get_task_2g, get_task_2h)


def main():
    # %% initialize the values
    x_bar = np.zeros(2)  # initial estimate
    P = 25 * np.eye(2)  # covariance of initial estimate


    z_c = np.array([2, 14])  # measurement 1
    R_c = np.array([[79, 36], [36, 36]])  # covariance of measurement 1
    H_c = np.eye(2)  # measurement matrix of measurement 2

    z_r = np.array([-4, 6])  # measurement 2
    R_r = np.array([[28, 4], [4, 22]])  # covariance of measurement 2
    H_r = np.eye(2)  # measurement matrix of measurement 1

    x_bar_c, P_c, x_bar_r, P_r = get_task_2f(x_bar, P,
                                             z_c, R_c, H_c,
                                             z_r, R_r, H_r)

    x_bar_rc, P_rc, x_bar_cr, P_cr = get_task_2g(x_bar_c, P_c,
                                                 x_bar_r, P_r,
                                                 z_c, R_c, H_c,
                                                 z_r, R_r, H_r)

    prob_above_line = get_task_2h(x_bar_rc, P_rc)
    print(f"Probability that it is above x_2 = x_1 + 5 is {prob_above_line}")

    interactive = InteractiveCovariance(condition_mean,
                                        condition_cov)

    show_task_2f_and_2g(x_bar, P,
                        z_c, R_c, x_bar_c, P_c,
                        z_r, R_r, x_bar_r, P_r,
                        x_bar_rc, P_rc,
                        x_bar_cr, P_cr)

    show_task_2h(x_bar_rc, P_rc)

    plt.show()


if __name__ == '__main__':
    main()
