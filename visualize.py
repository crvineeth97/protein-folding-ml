import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    def __init__(self):
        self.is_plot_initialized = False
        plt.ion()
        self.fig = plt.figure(figsize=(16, 9))
        self.ax = self.fig.add_subplot(111)
        plt.title("Ramachandran plot")
        ticks = np.arange(-180, 181, 30)
        self.ax.set_xticks(ticks)
        self.ax.set_yticks(ticks)
        self.ax.grid(which="both")
        self.ax.set_ylabel("Psi")
        self.ax.set_xlabel("Phi", color="black")
        self.pred_line = None
        self.act_line = None

    def plot_ramachandran(
        self, pred_phi, pred_psi, act_phi, act_psi, phi_mae=None, psi_mae=None
    ):
        pred_phi *= 180.0 / np.pi
        pred_psi *= 180.0 / np.pi
        act_phi *= 180.0 / np.pi
        act_psi *= 180.0 / np.pi
        if not self.is_plot_initialized:
            self.pred_line, = self.ax.plot(pred_phi, pred_psi, "ro", markersize=2.5)
            self.act_line, = self.ax.plot(act_phi, act_psi, "bo", markersize=2)
            self.text = self.ax.text(
                1,
                1,
                str(phi_mae) + " | " + str(psi_mae),
                horizontalalignment="right",
                verticalalignment="bottom",
                transform=self.ax.transAxes,
            )
            self.is_plot_initialized = True
        else:
            self.pred_line.set_xdata(pred_phi)
            self.pred_line.set_ydata(pred_psi)
            self.act_line.set_xdata(act_phi)
            self.act_line.set_ydata(act_psi)
            self.text.set_text(str(phi_mae) + " | " + str(psi_mae))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
