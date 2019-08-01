import torch
import time
import numpy as np
from os import listdir
from dataloader import contruct_dataloader_from_disk
from constants import TESTING_FOLDER, MINIBATCH_SIZE
from training import init_plot
from preprocessing import filter_input_files


def compute_mae(lengths, pred, act):
    mae = 0
    for i in range(MINIBATCH_SIZE):
        tmp = 0
        for j in range(lengths[i]):
            tmp += abs(pred[i][j] - act[i][j])
        tmp /= lengths[i]
        mae += tmp
    mae /= MINIBATCH_SIZE
    return mae


def test_model(path, criterion):
    model = torch.load(path, map_location={"cuda:0": "cpu"})
    is_plt_initialized = False
    loss = 0
    phi_mae = 0
    psi_mae = 0
    with torch.no_grad():
        test_loader = contruct_dataloader_from_disk(TESTING_FOLDER)
        test_size = test_loader.dataset.__len__()
        for i, data in enumerate(test_loader):
            # Tertiary is [Batch, Length, 9]
            lengths, primary, evolutionary, act_phi, act_psi, act_omega, tertiary = data
            inp = model.generate_input(lengths, primary, evolutionary)
            # Doesn't require gradients to go backwards, hence detach the output
            target = model.generate_target(lengths, act_phi, act_psi, act_omega)
            output = model(inp)
            loss += model.calculate_loss(lengths, criterion, output, target).item()
            # The following will be of size [Batch, Length]
            output = output.cpu().numpy()
            pred_phi = np.arctan2(output[:, 0, :], output[:, 1, :]) * 180.0 / np.pi
            pred_psi = np.arctan2(output[:, 2, :], output[:, 3, :]) * 180.0 / np.pi
            phi_mae += compute_mae(lengths, pred_phi, act_phi)
            psi_mae += compute_mae(lengths, pred_psi, act_psi)
            if not is_plt_initialized:
                fig, ax = init_plot()
                pred_line, act_line, = ax.plot(
                    pred_phi[0], pred_psi[0], "ro", act_phi[0], act_psi[0], "bo"
                )
                is_plt_initialized = True
            else:
                pred_line.set_xdata(pred_phi[0])
                pred_line.set_ydata(pred_psi[0])
                act_line.set_xdata(act_phi[0])
                act_line.set_ydata(act_psi[0])
                fig.canvas.draw()
                fig.canvas.flush_events()
            # time.sleep(5)
        loss /= test_size
        phi_mae /= test_size
        psi_mae /= test_size
    return loss, phi_mae, psi_mae


criterion = torch.nn.MSELoss()
input_files = listdir("output/")
input_files_filtered = filter_input_files(input_files)
for model_dir in input_files_filtered:
    print("Testing model " + model_dir)
    model_path = "output/" + model_dir + "/best.model"
    loss, phi_mae, psi_mae = test_model(model_path, criterion)
    print("Testing loss: " + str(loss))
    print("Phi MAE: " + str(phi_mae))
    print("Psi MAE: " + str(psi_mae))
