import logging
import time
from os import listdir

import numpy as np
import torch

from constants import HIDE_UI, TESTING_FOLDER
from dataloader import contruct_dataloader_from_disk
from preprocessing import filter_input_files
from util import get_model_dir
from visualize import Visualizer


def compute_mae(length, pred, act):
    # Compute MAE in degrees
    mae = 0
    tmp = 0
    for j in range(length):
        difference = abs(
            np.arctan2(np.sin(act[j] - pred[j]), np.cos(act[j] - pred[j]))
            * 180.0
            / np.pi
        )
        tmp += difference
    tmp /= length
    mae += tmp
    return mae


def test_model(model, criterion, model_dir=None, sleep_time=0):
    loss = 0
    running_phi_mae = 0
    running_psi_mae = 0
    visualize = None
    with torch.no_grad():
        test_loader = contruct_dataloader_from_disk(TESTING_FOLDER, 1)
        test_size = test_loader.dataset.__len__()
        for i, data in enumerate(test_loader):
            # Tertiary is [Batch, Length, 9]
            lengths, primary, evolutionary, act_phi, act_psi, act_omega, tertiary = data
            inp = model.generate_input(lengths, primary, evolutionary)
            # Doesn't require gradients to go backwards, hence detach the output
            target = model.generate_target(lengths, act_phi, act_psi, act_omega)
            output = model(inp, lengths)
            loss += model.calculate_loss(lengths, criterion, output, target).item()
            # The following will be of size [Batch, Length]
            output = output.cpu().numpy()
            pred_phi = np.arctan2(output[:, 0, :], output[:, 1, :])
            pred_psi = np.arctan2(output[:, 2, :], output[:, 3, :])
            phi_mae = compute_mae(lengths[0], pred_phi[0], act_phi[0])
            psi_mae = compute_mae(lengths[0], pred_psi[0], act_psi[0])
            running_phi_mae += phi_mae
            running_psi_mae += psi_mae
            if not HIDE_UI:
                if not visualize:
                    visualize = Visualizer()
                visualize.plot_ramachandran(
                    pred_phi[0], pred_psi[0], act_phi[0], act_psi[0], phi_mae, psi_mae
                )
                time.sleep(sleep_time)
        loss /= test_size
        running_phi_mae /= test_size
        running_psi_mae /= test_size

    if model_dir:
        from sys import stdout

        write_to = "output/" + model_dir + "/testing.txt"
        logging.basicConfig(stream=stdout, level=logging.INFO)
    else:
        write_to = get_model_dir() + "summary.txt"

    logging.info("Testing loss: %.10lf", loss)
    logging.info("Phi MAE: %.10lf", phi_mae)
    logging.info("Psi MAE: %.10lf", psi_mae)

    with open(write_to, "a") as f:
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
        test_model(model, criterion, model_dir, 15)
