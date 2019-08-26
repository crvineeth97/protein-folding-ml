import logging

import numpy as np
import torch

import pnerf.pnerf as pnerf
from constants import (
    DEVICE,
    EVAL_INTERVAL,
    HIDE_UI,
    MINIBATCH_SIZE,
    PRINT_LOSS_INTERVAL,
    TRAINING_EPOCHS,
    TRAINING_FOLDER,
    VALIDATION_FOLDER,
)
from dataloader import contruct_dataloader_from_disk
from utils import (
    calculate_rmsd,
    calculate_loss,
    generate_input,
    generate_target,
    get_model_dir,
    write_model_to_disk,
)
from visualize import Visualizer


def transform_tertiary(lengths, tertiary):
    batch_size = len(lengths)
    trans_tert = torch.zeros(
        batch_size, lengths[0] * 3, 3, device=DEVICE, dtype=torch.float32
    )
    for i in range(batch_size):
        for j in range(lengths[i]):
            trans_tert[i, 3 * j + 0] = torch.from_numpy(tertiary[i][j, 0:3])
            trans_tert[i, 3 * j + 1] = torch.from_numpy(tertiary[i][j, 3:6])
            trans_tert[i, 3 * j + 2] = torch.from_numpy(tertiary[i][j, 6:9])
    return trans_tert


def validate_model(model, criterion):
    with torch.no_grad():
        validation_loader = contruct_dataloader_from_disk(VALIDATION_FOLDER)
        validation_size = len(validation_loader)
        running_loss = 0.0
        running_rmsd = 0.0
        for _, data in enumerate(validation_loader):
            # Tertiary is [Batch, Length, 9]
            lengths, primary, evolutionary, act_phi, act_psi, act_omega, tertiary = data
            batch_size = len(lengths)
            inp = generate_input(lengths, primary, evolutionary)
            # Doesn't require gradients to go backwards, hence detach the output
            target = generate_target(lengths, act_phi, act_psi, act_omega)
            output = model(inp, lengths)
            batch_loss = calculate_loss(lengths, criterion, output, target).item()
            running_loss += batch_loss
            # The following will be of size [Batch, Length]
            pred_phi = torch.atan2(output[:, 0, :], output[:, 1, :]).unsqueeze(1)
            pred_psi = torch.atan2(output[:, 2, :], output[:, 3, :]).unsqueeze(1)
            # pred_omega = torch.atan2(output[:, 4, :], output[:, 5, :]).unsqueeze(1)
            pred_omega = torch.zeros(
                batch_size, 1, lengths[0], device=DEVICE, dtype=torch.float32
            )
            for j in range(batch_size):
                pred_omega[j, 0, : lengths[j]] = torch.from_numpy(act_omega[j])
            # dihedrals will be converted from [Batch, 3, length]
            # [Length, Batch, 3] as this is the input
            # required by pnerf functions
            dihedrals = torch.cat((pred_phi, pred_psi, pred_omega), 1).permute(2, 0, 1)
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
            predicted_coords = coordinates.transpose(0, 1)
            actual_coords = transform_tertiary(lengths, tertiary)
            batch_rmsd = calculate_rmsd(predicted_coords, actual_coords, lengths)
            running_rmsd += batch_rmsd
        running_loss = (
            (running_loss - batch_loss) * MINIBATCH_SIZE + batch_loss * batch_size
        ) / validation_size
        running_rmsd = (
            (running_rmsd - batch_rmsd) * MINIBATCH_SIZE + batch_rmsd * batch_size
        ) / validation_size
    return running_loss, running_rmsd


def train_model(model, criterion, optimizer):
    train_loader = contruct_dataloader_from_disk(TRAINING_FOLDER)
    train_size = len(train_loader.dataset)

    best_model_val_loss = 1e20
    best_model_rmsd = 1e20
    best_rmsd = 1e20
    visualize = None
    epoch_iter = 0
    no_of_batches = train_size // MINIBATCH_SIZE + 1

    while epoch_iter < TRAINING_EPOCHS:
        epoch_train_loss = 0.0
        running_train_loss = 0.0
        for batch_iter, data in enumerate(train_loader):
            # data is a tuple of tuple of numpy arrays except
            # lengths, which is a tuple of ints
            # Implies primary will be a tuple with MINIBATCH_SIZE number of elements
            # And each element will be of shape (Length,)
            # phi, psi and omega are in radians here
            lengths, primary, evolutionary, act_phi, act_psi, act_omega = data
            batch_size = len(lengths)
            # inp should be of shape [Batch, 41, Max_length]
            inp = generate_input(lengths, primary, evolutionary)
            # target should be of shape [Batch, 4, Max_length]
            target = generate_target(lengths, act_phi, act_psi, act_omega)
            # output should be of shape [Batch, 4, Max_length]
            # sin(phi), cos(phi), sin(psi), cos(psi), sin(omega), cos(omega)
            output = model(inp, lengths)
            batch_loss = calculate_loss(lengths, criterion, output, target)
            epoch_train_loss += batch_loss.item()
            running_train_loss += batch_loss.item()
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            if (batch_iter + 1) % PRINT_LOSS_INTERVAL == 0:
                logging.info(
                    "[%d|%.2f%%] Train loss: %.10lf",
                    epoch_iter,
                    ((batch_iter + 1) / no_of_batches) * 100,
                    running_train_loss / PRINT_LOSS_INTERVAL,
                )
                running_train_loss = 0.0

            if batch_iter % EVAL_INTERVAL == 0:
                # Finding validation loss in the beginning just to see how much
                # the model improves after time
                val_loss, rmsd = validate_model(model, criterion)
                if rmsd < best_rmsd:
                    best_rmsd = rmsd
                if val_loss < best_model_val_loss:
                    best_model_val_loss = val_loss
                    best_model_rmsd = rmsd
                    write_model_to_disk(model, "best")
                if not HIDE_UI:
                    output = output.detach().cpu().numpy()[0]
                    pred_phi = np.arctan2(output[0, :], output[1, :])
                    pred_psi = np.arctan2(output[2, :], output[3, :])
                    if not visualize:
                        visualize = Visualizer()
                    visualize.plot_ramachandran(
                        pred_phi, pred_psi, act_phi[0], act_psi[0]
                    )
                logging.info("\tValidation loss: %.10lf, RMSD: %.10lf", val_loss, rmsd)
        epoch_train_loss = (
            (epoch_train_loss - batch_loss) * MINIBATCH_SIZE + batch_loss * batch_size
        ) / train_size
        logging.info("Epoch train loss: %.10lf", epoch_train_loss)
        write_model_to_disk(model, "latest")
        epoch_iter += 1

    with open(get_model_dir() + "summary.txt", "a") as f:
        f.write("Number of epochs: " + str(epoch_iter) + "\n")
        f.write("Latest epoch train loss: " + str(epoch_train_loss) + "\n")
        f.write("Best model validation loss: " + str(best_model_val_loss) + "\n")
        f.write("Best model RMSD: " + str(best_model_rmsd) + "\n")
        f.write("Best RMSD: " + str(best_rmsd) + "\n")

    logging.info("Best model validation loss: %.10lf", best_model_val_loss)
    logging.info("Best model RMSD: %.10lf", best_model_rmsd)
    logging.info("Best RMSD: %.10lf", best_rmsd)
