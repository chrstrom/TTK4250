import numpy as np
from matplotlib.artist import Artist
import matplotlib.pyplot as plt


class InteractiveCovariance:

    showverts = True
    epsilon = 5  # max pixel distance to count as a vertex hit

    def __init__(self, condition_mean, condition_cov):
        self.condition_mean = condition_mean
        self.condition_cov = condition_cov

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_title(
            "Interactive plot to get an intuitive feeling of gaussians\n"
            'Drag blue or orange point to adjust estimate,\n'
            'drag black points or scroll to adjust covariance')

        self.fig.tight_layout()
        self.ax.set_xlim((-4, 4))
        self.ax.set_ylim((-3, 3))
        self.ax.set_aspect('equal')

        self.ellipse_points = np.array([[-1, 0],
                                        [-0.5, 0],
                                        [-1, 0.5],
                                        [1, 0],
                                        [1.5, 0],
                                        [1, 0.5]])
        self.circle_points = np.array([[np.cos(x), np.sin(x)]
                                       for x in np.linspace(0, 2*np.pi, 91)]).T

        self.active_vert = None  # the active vert
        self.showverts = True

        self.artist = []
        self.pred_scatter = self.ax.scatter([0], [0], s=10, c='g',
                                            animated=True)
        self.ellipse_scatter = self.ax.scatter(*self.ellipse_points.T,
                                               s=10,
                                               c=[*'bkk']+['orange']+[*'kk'],
                                               animated=True)
        self.ellipse_plots = [self.ax.plot(*points, animated=True)[0]
                              for points in self.get_ellipse_points()]

        self.canvas = self.fig.canvas
        self.draw_event_cid = self.canvas.mpl_connect(
            'draw_event', self.cb_draw)
        self.canvas.mpl_connect('button_press_event', self.cb_button_press)
        self.canvas.mpl_connect('button_release_event', self.cb_button_release)
        self.canvas.mpl_connect('motion_notify_event', self.cb_mouse_move)
        self.canvas.mpl_connect('scroll_event', self.cb_scroll)

    def get_ellipse_points(self):
        mat1 = self.ellipse_points[1:3].T - self.ellipse_points[0][:, None]
        mat2 = self.ellipse_points[4:6].T - self.ellipse_points[3][:, None]

        x = self.ellipse_points[0]
        z = self.ellipse_points[3]
        P = mat1@mat1.T
        R = mat2@mat2.T
        H = np.eye(2)

        x_hat = self.condition_mean(x, P, z, R, H)
        P_hat = self.condition_cov(P, R, H)

        self.pred_scatter.set_offsets(x_hat)
        return (mat1 @ self.circle_points + self.ellipse_points[0][:, None],
                mat2 @ self.circle_points + self.ellipse_points[3][:, None],
                (np.linalg.cholesky(P_hat) @ self.circle_points
                 + x_hat[:, None]))

    def cb_draw(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.draw(False)

    def draw(self, blit=True):
        self.canvas.restore_region(self.background)

        for plot, points in zip(self.ellipse_plots,
                                self.get_ellipse_points()):
            plot.set_data(*points)

            self.ax.draw_artist(plot)
        self.ax.draw_artist(self.ellipse_scatter)
        self.ax.draw_artist(self.pred_scatter)

        if blit:
            self.canvas.blit(self.ax.bbox)

    def poly_changed(self, poly):
        """This method is called whenever the pathpatch object is called."""
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        Artist.update_from(self.line, poly)
        self.line.set_visible(vis)  # don't use the poly visibility state

    def get_ind_under_point(self, event):
        """
        Return the index of the point closest to the event position or *None*
        if no point is within ``self.epsilon`` to the event position.
        """
        # display coords
        xyt = self.ax.transData.transform(self.ellipse_points)
        diff = xyt - np.array([[event.x, event.y]])
        closest = np.argmin(np.linalg.norm(diff, axis=1))
        return closest

    def cb_button_press(self, event):
        """Callback for mouse button presses."""
        if not self.showverts or event.inaxes is None or event.button != 1:
            return
        self.active_vert = self.get_ind_under_point(event)

    def cb_button_release(self, event):
        """Callback for mouse button releases."""
        if not self.showverts or event.inaxes is None or event.button != 1:
            return
        self.active_vert = None

    def cb_mouse_move(self, event):
        """Callback for mouse movements."""

        if (not self.showverts
                or event.inaxes is None
                or self.active_vert is None):
            return

        mouse_pos = np.array([event.xdata, event.ydata])

        scatterdata = self.ellipse_points
        group = self.active_vert // 3
        point = self.active_vert % 3
        affected = scatterdata[group*3: group*3+3]

        if point == 0:
            affected += (
                mouse_pos - scatterdata[self.active_vert, :])[None, :]

        if point == 1:
            affected[1, :] = mouse_pos
            veca = affected[1] - affected[0]
            vecb = affected[2] - affected[0]
            vecb = (np.array([[0, -1], [1, 0]])@veca
                    * np.linalg.norm(vecb) / np.linalg.norm(veca))
            affected[2, :] = affected[0, :] + vecb

        if point == 2:
            affected[2, :] = mouse_pos
            veca = affected[1] - affected[0]
            vecb = affected[2] - affected[0]
            veca = (np.array([[0, 1], [-1, 0]])@vecb
                    * np.linalg.norm(veca) / np.maximum(np.linalg.norm(vecb),
                                                        0.01))
            affected[1, :] = affected[0, :] + veca

        self.ellipse_scatter.set_offsets(scatterdata)

        self.draw()

    def cb_scroll(self, event):
        gxy = self.ax.transData.transform(self.ellipse_points)
        diff = gxy[::3] - np.array([[event.x, event.y]])
        group = np.argmin(np.linalg.norm(diff, axis=1))
        scatterdata = self.ellipse_points
        affected = scatterdata[group*3: group*3+3]

        scaling = 1.1**event.step
        affected[1] = scaling*(affected[1] - affected[0]) + affected[0]
        affected[2] = scaling*(affected[2] - affected[0]) + affected[0]
        self.ellipse_scatter.set_offsets(scatterdata)

        self.draw()


if __name__ == '__main__':

    p = InteractiveCovariance()
    plt.show()
