import numpy as np
from numpy import linalg as nla
import matplotlib.pyplot as plt
from utils.multivargaussian import MultiVarGaussian
import matplotlib as mpl

from pdaf import PDAF

mpl.rcParams['keymap.back'].remove('left')
mpl.rcParams['keymap.forward'].remove('right')

c_gt = 'green'
c_measurement = 'blue'
c_z_true = 'cyan'
c_estimate = 'red'
c_cov = 'orange'
c_gate = 'purple'


def get_ellipse_points(gaussian: MultiVarGaussian, scale=1):
    t = np.linspace(0, 2*np.pi, 91)
    circle_points = np.array([np.cos(t), np.sin(t)]) * scale

    mean, cov = gaussian
    lower = nla.cholesky(cov)
    return lower@circle_points + mean[:, None]


class InteractivePlot:
    def __init__(self,
                 pdaf: PDAF,
                 pos_gt_data,
                 pos_upd_gauss_seq,
                 measurement_pred_gauss_seq,
                 measurement_seq,
                 association_gt_seq,
                 pos_RMSE, vel_RMSE):
        self.pdaf = pdaf
        self.pos_gt_data = pos_gt_data
        self.pos_upd_gauss_seq = pos_upd_gauss_seq
        self.measurement_pred_gauss_seq = measurement_pred_gauss_seq

        self.measurement_seq = measurement_seq
        self.association_gt_seq = association_gt_seq

        self.gate_scaling = np.sqrt(self.pdaf.gate_size_sq)

        self.cur_idx = 0
        self.cur_state_gauss = self.pos_upd_gauss_seq[self.cur_idx]
        self.cur_z_pred_gauss = self.measurement_pred_gauss_seq[self.cur_idx]

        self.max_len_meas = max(len(zs) for zs in measurement_seq)
        self.cur_meas = np.empty((self.max_len_meas, 2))
        self.cur_meas[:] = np.nan
        cur_meas = self.measurement_seq[self.cur_idx]
        self.cur_meas[:len(cur_meas)] = cur_meas

        self.pos_history = self.cur_state_gauss.mean[None, :]
        self.pos_gt_history = self.pos_gt_data[self.cur_idx][None, :]
        self.paused = True
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_title("Trajectory visualization\n"
                          f"$RMSE_{{pos}}={pos_RMSE:.2f}m$, "
                          f"$RMSE_{{vel}}={vel_RMSE:.2f}m/s$\n"
                          "Controls: space=play, arrows=step, r=reset")

        self.ax.set_xlim((0, 700))
        self.ax.set_ylim((-100, 300))
        self.ax.set_autoscale_on(False)
        self.ax.set_aspect('equal')
        self.step_label = plt.plot([], [], ' ', label=f"K={0: 3d}")[0]

        self.pos_history_plot = self.ax.plot(
            [], [], c=c_estimate, marker='.', markersize=5,
            label=r"$\mathbf{\hat x}_{0:k}$", animated=True)[0]

        self.state_cov_plot = self.ax.plot(
            *get_ellipse_points(self.cur_state_gauss),
            c=c_cov, label=r"$\mathbf{P}_k$", animated=True)[0]

        self.gate_plot = self.ax.plot(
            *get_ellipse_points(self.cur_z_pred_gauss, self.gate_scaling),
            c=c_gate, ls=':', animated=True, label="gate")[0]

        self.pos_gt_history_plot = self.ax.plot(
            [], [], c=c_gt, marker='.', markersize=5,
            label=r"$\mathbf{x}_{gt 0:k}$", animated=True)[0]

        self.measurement_scatter = self.ax.scatter(
            *self.cur_meas.T, c=c_measurement, s=10,
            label=r"$\mathbf{Z}_k$ (cyan is correct)", animated=True)

        self.legend = self.ax.legend(loc='upper left', framealpha=1)
        self.fig.tight_layout()

        self.canvas = self.fig.canvas
        self.timer = self.canvas.new_timer(100)
        self.timer.add_callback(self.cb_timer)
        self.timer.start()

        self.draw_event_cid = self.canvas.mpl_connect('draw_event',
                                                      self.cb_fullraw)
        self.canvas.mpl_connect('key_press_event', self.cb_key_press)

    def cb_fullraw(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.draw(False)

    def draw(self, blit=True):
        self.cur_state_gauss = self.pos_upd_gauss_seq[self.cur_idx]
        self.cur_z_pred_gauss = self.measurement_pred_gauss_seq[self.cur_idx]

        self.pos_history = np.array([
            state.mean for state in self.pos_upd_gauss_seq[:self.cur_idx+1]]).T
        cur_meas = self.measurement_seq[self.cur_idx]

        self.cur_meas[:] = np.nan
        self.cur_meas[:len(cur_meas)] = cur_meas

        self.canvas.restore_region(self.background)
        self.step_label.set_label(f"K={self.cur_idx: 3d}")
        self.legend = self.ax.legend(loc='upper left', framealpha=1)
        self.state_cov_plot.set_data(*get_ellipse_points(self.cur_state_gauss))
        self.gate_plot.set_data(*get_ellipse_points(self.cur_z_pred_gauss,
                                                    self.gate_scaling))
        self.pos_history_plot.set_data(*self.pos_history)
        self.pos_gt_history_plot.set_data(*self.pos_gt_data[:self.cur_idx].T)

        self.measurement_scatter.set_offsets(self.cur_meas)
        colors = [c_measurement]*self.max_len_meas
        if self.association_gt_seq[self.cur_idx]:
            colors[self.association_gt_seq[self.cur_idx]-1] = c_z_true
        self.measurement_scatter.set_color(colors)

        self.ax.draw_artist(self.state_cov_plot)
        self.ax.draw_artist(self.gate_plot)
        self.ax.draw_artist(self.pos_history_plot)
        self.ax.draw_artist(self.pos_gt_history_plot)
        self.ax.draw_artist(self.measurement_scatter)
        self.ax.draw_artist(self.legend)

        if blit:
            self.canvas.blit(self.ax.bbox)

    def cb_timer(self):
        if not self.paused:
            self.cur_idx = min(self.cur_idx+1, len(self.pos_upd_gauss_seq)-1)
            self.draw()

    def cb_key_press(self, event):

        self.cur_state_gauss
        if event.key == 'right':
            self.cur_idx = (self.cur_idx+1) % len(self.pos_upd_gauss_seq)
        elif event.key == 'left':
            self.cur_idx = (self.cur_idx-1) % len(self.pos_upd_gauss_seq)

        if event.key == 'up':
            self.cur_idx = (self.cur_idx+10) % len(self.pos_upd_gauss_seq)
        elif event.key == 'down':
            self.cur_idx = (self.cur_idx-10) % len(self.pos_upd_gauss_seq)

        elif event.key == ' ':
            self.paused = self.paused ^ True
        elif event.key == 'r':
            self.ax.set_xlim((0, 700))
            self.ax.set_ylim((-100, 300))
            self.ax.set_aspect('equal')

            self.cur_idx = 0

        self.draw()
