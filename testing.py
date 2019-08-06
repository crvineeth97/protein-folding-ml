import logging
from os import listdir

import numpy as np
import torch

from constants import MINIBATCH_SIZE, TESTING_FOLDER, HIDE_UI
from dataloader import contruct_dataloader_from_disk
from preprocessing import filter_input_files
from visualize import Visualizer
from util import get_model_dir


def compute_mae(lengths, pred, act):
    # Compute MAE in degrees
    mae = 0
    for i in range(MINIBATCH_SIZE):
        tmp = 0
        for j in range(lengths[i]):
            tmp += abs(pred[i][j] - act[i][j]) * 180.0 / np.pi
        tmp /= lengths[i]
        mae += tmp
    mae /= MINIBATCH_SIZE
    return mae


def test_model(model, criterion, sleep_time=0):
    visualize = Visualizer()
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
            pred_phi = np.arctan2(output[:, 0, :], output[:, 1, :])
            pred_psi = np.arctan2(output[:, 2, :], output[:, 3, :])
            phi_mae += compute_mae(lengths, pred_phi, act_phi)
            psi_mae += compute_mae(lengths, pred_psi, act_psi)
            if not HIDE_UI:
                visualize.plot_ramachandran(
                    pred_phi[0], pred_psi[0], act_phi[0], act_psi[0]
                )
        loss /= test_size
        phi_mae /= test_size
        psi_mae /= test_size
    logging.info("Testing loss: %f", loss)
    logging.info("Phi MAE: %f", phi_mae)
    logging.info("Psi MAE: %f", psi_mae)

    with open(get_model_dir() + "summary.txt", "a") as f:
        f.write("Testing loss: " + str(loss) + "\n")
        f.write("Phi MAE: " + str(phi_mae) + "\n")
        f.write("Psi MAE: " + str(psi_mae) + "\n")


if __name__ == "__main__":
    criterion = torch.nn.MSELoss()
    input_files = listdir("output/")
    input_files_filtered = filter_input_files(input_files)
    for model_dir in input_files_filtered:
        print("Testing model " + model_dir)
        model_path = "output/" + model_dir + "/best.model"
        model = torch.load(model_path, map_location={"cuda:0": "cpu"})
        test_model(model, criterion, 5)
