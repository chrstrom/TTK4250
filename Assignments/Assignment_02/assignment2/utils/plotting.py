
import matplotlib as mpl

import matplotlib.pyplot as plt

from utils.plot_ellipse import plot_cov_ellipse2d

mpl.use('Qt5Agg')  # needs the pyqt package,
# crazy cpu usage sometimes but better behaved than MacOSX
# to see your plot config
print(f"matplotlib backend: {mpl.get_backend()}")
print(f"matplotlib config file: {mpl.matplotlib_fname()}")
print(f"matplotlib config dir: {mpl.get_configdir()}")
# installed with "pip install SciencePLots" (https://github.com/garrettj403/SciencePlots.git)
# plt.style.use(['science', 'grid', 'ieee', 'bright']) # gives quite nice plots

plt.close("all")


def show_task_2f_and_2g(x_bar, P,
                        z_c, R_c, x_bar_c, P_c,
                        z_r, R_r, x_bar_r, P_r,
                        x_bar_rc, P_rc,
                        x_bar_cr, P_cr):

    fig, ax = plt.subplots()
    ax.set_title("Task 2f and 2g")

    plot_cov_ellipse2d(ax, x_bar, P, edgecolor="C0")
    ax.scatter(*x_bar, c="C0", marker="x", label=r"$\bar x$")

    plot_cov_ellipse2d(ax, x_bar_c, P_c, edgecolor="C1")
    ax.scatter(*x_bar_c, c="C1", marker="x", label=r"$\bar x_c$")

    plot_cov_ellipse2d(ax, x_bar_r, P_r, edgecolor="C2")
    ax.scatter(*x_bar_r, c="C2", marker="x", label=r"$\bar x_r$")

    plot_cov_ellipse2d(ax, x_bar_cr, P_cr, edgecolor="C3")
    ax.scatter(*x_bar_cr, c="C3", marker="x", label=r"$\bar x_{cr}$")

    plot_cov_ellipse2d(ax, x_bar_rc, P_rc, edgecolor="cyan", linestyle="--")
    ax.scatter(*x_bar_rc, c="cyan", marker="+", label=r"$\bar x_{rc}$")

    # %% measurements
    ax.scatter(*z_c, c="C1", label="$z_c$")
    plot_cov_ellipse2d(ax, z_c, R_c, edgecolor="C1")

    ax.scatter(*z_r, c="C2", label="$z_r$")
    plot_cov_ellipse2d(ax, z_r, R_r, edgecolor="C2")
    # % true value
    # ax.scatter(-5, 12, c="C6", marker="^", label="$x$")

    ax.axis("equal")
    ax.legend()

    plt.show(block=False)


def show_task_2h(x_bar_rc, P_rc):
    fig, ax = plt.subplots()
    ax.set_title("Task 2h")

    plot_cov_ellipse2d(ax, x_bar_rc, P_rc, edgecolor="C0")
    ax.scatter(*x_bar_rc, marker="x", c="C0", label=r"$\bar x_{rc}$")
    ax.plot([-1, 4], [4, 9], color="C1", label="$x_2 = x_1 + 5$")

    ax.axis("equal")
    ax.legend()
    plt.show(block=False)
