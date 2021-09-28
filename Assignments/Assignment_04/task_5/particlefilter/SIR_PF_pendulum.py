# %% imports
from typing import Callable, Final
import solution
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import zeros_like
import scipy.stats

# sets the seed for random numbers to be predictable
DEBUG: Final[bool] = True  # TODO: set to false when working


def bind_variables(func, **kwargs):
    def func_bind(*args, **kwargs_inn):
        kwargs_upd = kwargs | kwargs_inn
        return func(*args, **kwargs_upd)
    return func_bind

# %% trajectory generation


def get_scenario_parameters():
    # scenario parameters
    x0 = np.array([np.pi / 2, -np.pi / 100])
    Ts = 0.05
    K = round(20 / Ts)
    return x0, Ts, K


def get_dynamic_parameters():
    # constants
    g = 9.81
    l = 1
    a = g / l
    d = 0.5  # dampening
    S = 5
    return l, a, d, S


def get_measurement_parameters():
    # TODO: change these in (b)
    Ld = 4
    Ll = 0
    r = 0.25
    return Ld, Ll, r

# disturbance PDF


def process_noise_sampler(rng, S): return rng.uniform(-S, S)

# dynamic function


def modulo2pi(x, idx=0):
    xmod = x
    xmod[idx] = (xmod[idx] + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
    return xmod


def pendulum_dynamics(x, a, d=0):  # continuous dynamics
    xdot = np.array([x[1], -d * x[1] - a * np.sin(x[0])])
    return xdot


def pendulum_dynamics_discrete(xk, vk, Ts, a, d=0):
    xkp1 = modulo2pi(xk + Ts * pendulum_dynamics(xk, a, d))  # euler discretize
    xkp1[1] += Ts * vk  # zero order hold noise
    return xkp1


# sample a trajectory
def sample_trajectory(x0, Ts, K, process_noise_sampler, pendulum_dynamics_discrete):
    x = np.zeros((K, 2))
    x[0] = x0
    for k in range(K - 1):
        v = process_noise_sampler()
        x[k + 1] = pendulum_dynamics_discrete(x[k], v, Ts)
    return x


# vizualize
def plot_trajectory(x):
    fig, axs = plt.subplots(2, sharex=True, num=1, clear=True)
    axs[0].plot(x[:, 0])
    axs[0].set_ylabel(r"$\theta$")
    axs[0].set_ylim((-np.pi, np.pi))

    axs[1].plot(x[:, 1])
    axs[1].set_xlabel("Time step")
    axs[1].set_ylabel(r"$\dot \theta$")
    return fig, axs


# %% measurement generation

# noise pdf


def measurement_noise_sampler(x, r, rng): return rng.triangular(-r, 0, r)

# measurement function

# makes a decorator that binds keyword arguments to a function


def measurement_function(x, Ld, l, Ll):
    lcth = l * np.cos(x[0])
    lsth = l * np.sin(x[0])
    z = np.sqrt((Ld - lcth) ** 2 + (lsth - Ll) ** 2)  # 2norm
    return z


def sample_measurements(x, h, measurement_noise_sampler):
    Z = np.zeros(len(x))
    for k, xk in enumerate(x):
        wk = measurement_noise_sampler(xk)
        Z[k] = h(xk) + wk
    return Z


# vizualize
def plot_measurements(Z, fignum):
    fig, ax = plt.subplots(num=fignum, clear=True)
    ax.plot(Z)
    ax.set_xlabel("Time step")
    ax.set_ylabel("z")
    return fig, ax


# %% Task: Estimate using a particle filter


def init_PF(rng):
    """initialize particles.

    Args:
        rng: a random number generator

    Returns:
        N (int): number of particles
        px (ndarray): particle states shape=(N, dim(state))
        weights (ndarray): normalized weights. shape = (N,)
    """
    # number of particles to use
    N = 10  # TODO

    # TODO: initialize particles, pretend you do not know where the pendulum starts
    px = np.array([None,  # Hint: use rng.somesampler
                   None]).T

    # initial weights
    w = np.zeros(N)  # TODO

    # TODO replace this with your own code
    N, px, w = solution.SIR_PF_pendulum.init_PF(rng)
    assert np.isclose(sum(w), 1), "w must be normalized"
    assert len(px) == N and len(w) == N, "number of particles must be N"

    return N, px, w


def weight_update(zk: float, px: np.ndarray, w: np.ndarray, h: Callable, meas_noise_dist: scipy.stats.distributions.rv_frozen):
    """Update the weights.

    Args:
        zk: measurement
        px: particles, shape = (N, dim(state))
        w: weights in, shape = (N, )
        h: heasurement funciton that takes the state as args.
        PF_measurement_distribution: the measurement distribution (a numpy random variable)

    Returns:
        updated_weights: shape = (N,) must be normalized
    """
    w_upd = np.empty_like(w)
    for n, pxn in enumerate(px):
        # TODO, hint: h, zk, pxn and PF_measurement_distribution.pdf
        w_upd[n] = 0
    w_upd = w_upd  # TODO: normalize

    # TODO replace this with your own code
    w_upd = solution.SIR_PF_pendulum.weight_update(
        zk, px, w, h, meas_noise_dist)

    return w_upd


def resample(px: np.ndarray, w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample particles

    Args:
        px: shape = (N, dim(state)), the particles
        w: shape = (N,), particle weights
        rng: random number generator.
            Must be used in order to be able to make predictable behaviour.

    Returns:
        pxn: the resampled particles
    """
    N = len(w)
    pxn = np.zeros_like(px)
    # TODO: some pre calculations?
    i = 0
    for n in range(N):
        # find a particle 'i' to pick
        # algorithm in the book, but there are other options as well
        i = i  # TODO
        pxn[n] = px[i]

    # TODO replace this with your own code
    pxn = solution.SIR_PF_pendulum.resample(px, w, rng)

    return pxn


def particle_prediction(px: np.ndarray, Ts: float, f: Callable, proc_noise_dist: scipy.stats.distributions.rv_frozen) -> np.ndarray:
    """Predict particles some time units ahead sampling the process noise.

    Args:
        px: shape = (N. dim(state)), the particles
        Ts: Time step size
        f: process function taking the state, noise realization and time step size as arguments
        dyn_dist: a distribution that can create process noise realizations

    Returns:
        px_pred: the predicted particles
    """
    px_pred = zeros_like(px)
    for n, pxn in enumerate(px):
        vkn = 0  # TODO: process noise realization, hint: proc_noise_dist.rvs
        px_pred[n] = 0  # TODO: particle prediction, hint: f

    # TODO replace this with your own code
    px_pred = solution.SIR_PF_pendulum.particle_prediction(
        px, Ts, f, proc_noise_dist)

    return px_pred


def plot_step(x, pxn, l, fig, plotpause, sch_particles, sch_true):
    sch_particles.set_offsets(
        np.c_[l * np.sin(pxn[:, 0]), -l * np.cos(pxn[:, 0])])
    sch_true.set_offsets(np.c_[l * np.sin(x[0]), -l * np.cos(x[0])])

    fig.canvas.draw_idle()
    plt.show(block=False)
    plt.waitforbuttonpress(plotpause)


def plot_setup_PF(l, fignum):
    plt.ion()
    fig, ax = plt.subplots(num=fignum, clear=True)

    sch_particles = ax.scatter(
        np.nan, np.nan, marker=".", c="b", label=r"$\hat \theta^n$")
    sch_true = ax.scatter(np.nan, np.nan, c="r", marker="x", label=r"$\theta$")
    ax.set_ylim((-1.5 * l, 1.5 * l))
    ax.set_xlim((-1.5 * l, 1.5 * l))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    th = ax.set_title(f"theta mapped to x-y")
    ax.legend()
    return fig, ax, sch_particles, sch_true


def run_SIR_PF(rng, Ts, l, f, x, h, Z, px, w, PF_dynamic_distribution, PF_measurement_distribution, plotpause, fignum=4):
    fig, ax, sch_particles, sch_true = plot_setup_PF(l, fignum=fignum)

    for k, zk in enumerate(Z):
        print(f"{k = }")

        w = weight_update(zk, px, w, h, PF_measurement_distribution)

        pxn = resample(px, w, rng)

        px = particle_prediction(
            pxn, Ts, f, PF_dynamic_distribution)

    # plot
        plot_step(x[k], pxn, l, fig, plotpause, sch_particles, sch_true)

    plt.waitforbuttonpress()
    # %%


def main():
    seed = 0 if DEBUG else None
    rng = np.random.default_rng(seed=seed)

    x0, Ts, K = get_scenario_parameters()
    l, a, d, S = get_dynamic_parameters()
    Ld, Ll, r = get_measurement_parameters()

    N, px, w = init_PF(rng)

    f = bind_variables(
        pendulum_dynamics_discrete, a=a, d=d)
    proc_sampler = bind_variables(process_noise_sampler, S=S, rng=rng)

    h = bind_variables(measurement_function, Ld=Ld, l=l, Ll=Ll)
    meas_sampler = bind_variables(measurement_noise_sampler, r=r, rng=rng)

    x = sample_trajectory(x0, Ts, K, proc_sampler, f)
    fig1, axs1 = plot_trajectory(x)

    Z = sample_measurements(x, h, meas_sampler)
    fig2, ax2 = plot_measurements(Z, 2)

    # PF transition PDF: SIR proposal, or something you would like to test
    PF_dynamic_distribution = scipy.stats.uniform(loc=-S, scale=2 * S)
    PF_measurement_distribution = scipy.stats.triang(
        c=0.5, loc=-r, scale=2 * r)

    # initialize a figure for particle animation.
    plotpause = 0.01

    run_SIR_PF(rng, Ts, l, f, x, h, Z, px, w, PF_dynamic_distribution,
               PF_measurement_distribution, plotpause, fignum=4)


if __name__ == "__main__":
    main()
