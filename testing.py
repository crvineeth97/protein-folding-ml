import torch
import time
import numpy as np
from os import listdir
from dataloader import contruct_dataloader_from_disk
from constants import TESTING_FOLDER
from training import init_plot
from preprocessing import filter_input_files


def test_model(path, criterion):
    model = torch.load(path)
    is_plt_initialized = False
    with torch.no_grad():
        test_loader = contruct_dataloader_from_disk(TESTING_FOLDER)
        test_size = test_loader.dataset.__len__()
        loss = 0
        for i, data in enumerate(test_loader):
            # Tertiary is [Batch, Length, 9]
            lengths, primary, evolutionary, act_phi, act_psi, act_omega, tertiary = data
            inp = model.generate_input(lengths, primary, evolutionary)
            # Doesn't require gradients to go backwards, hence detach the output
            target = model.generate_target(lengths, act_phi, act_psi, act_omega)
            output = model(inp)
            loss += model.calculate_loss(lengths, criterion, output, target)
            # The following will be of size [Batch, Length]
            output = output.cpu().numpy()[0]
            pred_phi = np.arctan2(output[0, :], output[1, :]) * 180.0 / np.pi
            pred_psi = np.arctan2(output[2, :], output[3, :]) * 180.0 / np.pi
            if not is_plt_initialized:
                fig, ax = init_plot()
                pred_line, act_line, = ax.plot(
                    pred_phi, pred_psi, "ro", act_phi[0], act_psi[0], "bo"
                )
                is_plt_initialized = True
            else:
                pred_line.set_xdata(pred_phi)
                pred_line.set_ydata(pred_psi)
                act_line.set_xdata(act_phi[0])
                act_line.set_ydata(act_psi[0])
                fig.canvas.draw()
                fig.canvas.flush_events()
            time.sleep(5)
        loss /= test_size
    return loss.item()


criterion = torch.nn.MSELoss()
input_files = listdir("output/models/")
input_files_filtered = filter_input_files(input_files)
for filename in input_files_filtered:
    print("Testing model " + filename)
    model_path = "output/models/" + filename
    loss = test_model(model_path, criterion)
    print("Testing loss: " + loss)
