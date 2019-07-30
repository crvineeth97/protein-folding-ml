import logging

import matplotlib.pyplot as plt
import numpy as np
import torch

import pnerf.pnerf as pnerf
from constants import (
    DEVICE,
    EVAL_INTERVAL,
    HIDE_UI,
    LEARNING_RATE,
    MINIBATCH_SIZE,
    TRAINING_EPOCHS,
    TRAINING_FOLDER,
    VALIDATION_FOLDER,
)
from dataloader import contruct_dataloader_from_disk
from util import calc_rmsd, write_model_to_disk


def init_plot():
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title("Ramachandran plot")
    ticks = np.arange(-180, 181, 30)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.grid(which="both")
    ax.set_ylabel("Psi")
    ax.set_xlabel("Phi", color="black")
    return fig, ax


def transform_tertiary(lengths, tertiary):
    trans_tert = torch.zeros(
        MINIBATCH_SIZE, lengths[0] * 3, 3, device=DEVICE, dtype=torch.float32
    )
    for i in range(MINIBATCH_SIZE):
        for j in range(lengths[i]):
            trans_tert[i, 3 * j + 0] = torch.from_numpy(tertiary[i][j, 0:3])
            trans_tert[i, 3 * j + 1] = torch.from_numpy(tertiary[i][j, 3:6])
            trans_tert[i, 3 * j + 2] = torch.from_numpy(tertiary[i][j, 6:9])
    return trans_tert


def validate_model(model, criterion):
    with torch.no_grad():
        validation_loader = contruct_dataloader_from_disk(VALIDATION_FOLDER)
        val_size = validation_loader.dataset.__len__()
        rmsd = 0
        loss = 0
        for i, data in enumerate(validation_loader):
            # Tertiary is [Batch, Length, 9]
            lengths, primary, evolutionary, act_phi, act_psi, act_omega, tertiary = data
            inp = model.generate_input(lengths, primary, evolutionary)
            # Doesn't require gradients to go backwards, hence detach the output
            target = model.generate_target(lengths, act_phi, act_psi, act_omega)
            output = model(inp)
            loss += model.calculate_loss(lengths, criterion, output, target)
            # The following will be of size [Batch, Length]
            pred_phi = torch.atan2(output[:, 0, :], output[:, 1, :]).unsqueeze(1)
            pred_psi = torch.atan2(output[:, 2, :], output[:, 3, :]).unsqueeze(1)
            pred_omega = torch.atan2(output[:, 4, :], output[:, 5, :]).unsqueeze(1)
            # dihedrals will be of size [Length, Batch, 3] as this is the input
            # required by pnerf functions
            dihedrals = (
                torch.cat((pred_phi, pred_psi, pred_omega), 1)
                .transpose(1, 2)
                .transpose(0, 1)
            )
            # Pnerf takes dihedrals and optional bond lengths and bond angles as input
            # and builds the coordinates so that rmsd can be calculated for loss
            # Divide by 100 to get in angstrom units
            # Coordinates will be of size [Length * 3, Batch, 3]
            # Last dimension represents the x, y, z coordinates
            # And the Length * 3 represents N, C_a, C for each AA
            coordinates = (
                pnerf.point_to_coordinate(
                    pnerf.dihedral_to_point(dihedrals, DEVICE), DEVICE
                )
                / 100
            )
            predicted_coords = coordinates.transpose(0, 1).contiguous().view(-1, 3)
            actual_coords = (
                transform_tertiary(lengths, tertiary).contiguous().view(-1, 3)
            )
            rmsd += calc_rmsd(predicted_coords, actual_coords)
            # drmsd += calc_drmsd(predicted_coords, actual_coords)
        rmsd /= val_size
        loss /= val_size
        # drmsd /= val_size
    return val_size, loss.item(), rmsd


def train_model(model):
    train_loader = contruct_dataloader_from_disk(TRAINING_FOLDER)
    lr = LEARNING_RATE
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    prev_model_val_loss = 1e20
    best_model_val_loss = 1e20
    is_plt_initialized = False
    best_model_path = None
    training_set_iter = 0

    while training_set_iter < TRAINING_EPOCHS:
        optimizer.zero_grad()
        model.zero_grad()
        pred_line = None
        act_line = None
        for i, data in enumerate(train_loader):
            # data is a tuple of tuple of tensors except
            # lengths, which is a tuple of ints
            # Implies primary will be a tuple with MINIBATCH_SIZE number of elements
            # And each element will be of shape (Length,)
            # phi, psi and omega are in degrees here
            lengths, primary, evolutionary, act_phi, act_psi, act_omega = data
            # inp should be of shape [Batch, 41, Max_length]
            inp = model.generate_input(lengths, primary, evolutionary)
            # target should be of shape [Batch, 6, Max_length]
            target = model.generate_target(lengths, act_phi, act_psi, act_omega)
            # output should be of shape [Batch, 6, Max_length]
            # sin(phi), cos(phi), sin(psi), cos(psi), sin(omega), cos(omega)
            output = model(inp)
            loss = model.calculate_loss(lengths, criterion, output, target)
            optimizer.zero_grad()
            logging.info("Train loss: %.10lf", loss.item())
            loss.backward()
            optimizer.step()

            if (i + 1) % EVAL_INTERVAL == 0:
                val_size, val_loss, rmsd = validate_model(model, criterion)
                if val_loss < best_model_val_loss:
                    prev_model_val_loss = best_model_val_loss
                    best_model_val_loss = val_loss
                    best_model_path = write_model_to_disk(model)
                    if (prev_model_val_loss - best_model_val_loss) < 0.001:
                        logging.info(
                            "Changing learning rate from %lf to %lf", lr, lr / 5
                        )
                        lr /= 5
                        for par in optimizer.param_groups:
                            par["lr"] = lr
                if not HIDE_UI:
                    output = output.detach().cpu().numpy()[0]
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
                logging.info(
                    "Validation loss: %.10lf Train loss: %.10lf", val_loss, loss.item()
                )
                logging.info(
                    "RMSD: %.10lf Best model stored at %s", rmsd, best_model_path
                )
        training_set_iter += 1

    return best_model_path
