import logging
import time
from os import listdir

import numpy as np
import torch

from constants import DEVICE, HIDE_UI, TESTING_FOLDER
from dataloader import contruct_dataloader_from_disk
from utils import calculate_loss, generate_input, generate_target
from visualize import Visualizer


def compute_mae(lengths, pred, act):
    # Compute MAE in degrees
    bs = len(lengths)
    mae = np.zeros(bs)
    for i in range(bs):
        for j in range(lengths[i]):
            diff = act[i][j] - pred[i][j]
            mae[i] += min(abs(diff), abs(2 * np.pi - diff)) * 180.0 / np.pi
        mae[i] /= lengths[i]
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
            inp = generate_input(lengths, primary, evolutionary)
            # Doesn't require gradients to go backwards, hence detach the output
            target = generate_target(lengths, act_phi, act_psi, act_omega)
            output = model(inp, lengths)
            loss += calculate_loss(lengths, criterion, output, target).item()
            # The following will be of size [Batch, Length]
            output = output.cpu().numpy()
            pred_phi = np.arctan2(output[:, 0, :], output[:, 1, :])
            pred_psi = np.arctan2(output[:, 2, :], output[:, 3, :])
            phi_mae = compute_mae(lengths, pred_phi, act_phi)
            psi_mae = compute_mae(lengths, pred_psi, act_psi)
            running_phi_mae += phi_mae.sum()
            running_psi_mae += psi_mae.sum()
            if not HIDE_UI:
                if not visualize:
                    visualize = Visualizer()
                bs = len(lengths)
                for i in range(bs):
                    visualize.plot_ramachandran(
                        pred_phi[i],
                        pred_psi[i],
                        act_phi[i],
                        act_psi[i],
                        phi_mae[i],
                        psi_mae[i],
                    )
                    time.sleep(sleep_time)
        # Fix loss w.r.t. batch_size
        loss /= test_size
        running_phi_mae /= test_size
        running_psi_mae /= test_size

    logging.info("Testing loss: %.10lf", loss)
    logging.info("Phi MAE: %.10lf", running_phi_mae)
    logging.info("Psi MAE: %.10lf", running_psi_mae)


if __name__ == "__main__":
    from sys import stdout
    from preprocess import filter_input_files

    stdout_handler = logging.StreamHandler(stdout)
    criterion = torch.nn.MSELoss()
    input_files = listdir("output/")
    input_files_filtered = filter_input_files(input_files)
    for model_dir in input_files_filtered:
        model_dir = "output/" + model_dir
        file_handler = logging.FileHandler(filename=model_dir + "testing.txt")
        handlers = [file_handler, stdout_handler]
        logging.basicConfig(
            format="%(asctime)s %(message)s",
            datefmt="%H:%M:%S",
            level=logging.INFO,
            handlers=handlers,
        )

        logging.info("Testing best model in " + model_dir)
        model_path = model_dir + "/best.model"
        if DEVICE == torch.device("cpu"):
            model = torch.load(model_path, map_location={"cuda:0": "cpu"})
        else:
            model = torch.load(model_path)
        test_model(model, criterion, model_dir, 1)

        logging.info("Testing latest model in " + model_dir)
        model_path = model_dir + "/latest.model"
        if DEVICE == torch.device("cpu"):
            model = torch.load(model_path, map_location={"cuda:0": "cpu"})
        else:
            model = torch.load(model_path)
        test_model(model, criterion, model_dir, 1)
