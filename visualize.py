import matplotlib.pyplot as plt
import numpy as np

plt.ion()


class RamachandranPlot:
    def __init__(self):
        self.is_plot_initialized = False
        self.fig = plt.figure(figsize=(16, 9))
        self.ax = self.fig.add_subplot(111)
        self.fig.suptitle("Ramachandran plot")
        ticks = np.arange(-180, 181, 30)
        self.ax.set_xticks(ticks)
        self.ax.set_yticks(ticks)
        self.ax.grid(which="both")
        self.ax.set_ylabel("Psi")
        self.ax.set_xlabel("Phi", color="black")
        self.pred_line = None
        self.act_line = None
        plt.pause(0.5)

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


class ContactMap:
    def __init__(self, use_binary=False):
        self.is_plot_initialized = False
        self.fig = plt.figure(figsize=(16, 9))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.img = None
        self.use_binary = use_binary
        plt.pause(0.5)

    def plot_contact_map(self, prot_id, contact_map):
        self.fig.suptitle(prot_id)
        if self.use_binary:
            contact_map = np.where(contact_map <= 8, 1, 0)
        # self.norm = norm = cm.colors.Normalize(
        #     vmax=abs(contact_map).max(), vmin=-abs(contact_map).max()
        # )
        if not self.is_plot_initialized:
            self.img = self.ax.imshow(contact_map, aspect="equal")
            self.is_plot_initialized = True
        else:
            self.img.set_data(A=contact_map)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
