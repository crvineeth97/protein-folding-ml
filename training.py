import numpy as np
import torch
import requests

import pnerf.pnerf as pnerf
from constants import (
    EVAL_INTERVAL,
    LEARNING_RATE,
    MIN_BATCH_ITER,
    MINIBATCH_SIZE,
    TRAINING_FOLDER,
    VALIDATION_FOLDER,
    HIDE_UI,
    DEVICE,
)
from dataloader import contruct_dataloader_from_disk
from util import (
    calc_rmsd,
    calc_drmsd,
    set_experiment_id,
    write_to_pdb,
    write_model_to_disk,
    write_out,
    write_result_summary,
)


def validate_model(model):
    validation_loader = contruct_dataloader_from_disk(VALIDATION_FOLDER)
    val_size = validation_loader.dataset.__len__()
    rmsd = 0
    drmsd = 0
    for i, data in enumerate(validation_loader):
        lengths, primary, evolutionary, phi, psi, omega, tertiary = data
        inp = model.generate_input(lengths, primary, evolutionary)
        output = model(inp)
        # The following will be of size [Batch, Length]
        phi = torch.atan2(output[:, 0, :], output[:, 1, :]).unsqueeze(1)
        psi = torch.atan2(output[:, 2, :], output[:, 3, :]).unsqueeze(1)
        omega = torch.atan2(output[:, 4, :], output[:, 5, :]).unsqueeze(1)
        # dihedrals will be of size [Length, Batch, 3] as this is the input
        # required by pnerf functions
        dihedrals = torch.cat((phi, psi, omega), 1).transpose(1, 2).transpose(0, 1)
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
        coords = coordinates.transpose(0, 1).contiguous().view(1, -1, 9).transpose(0, 1)
        predicted_coords = coords.transpose(0, 1).contiguous().view(-1, 3)
        actual_coords = (
            torch.from_numpy(tertiary)
            .unsqueeze(1)
            .transpose(0, 1)
            .contiguous()
            .view(-1, 3)
        )
        rmsd += calc_rmsd(predicted_coords, actual_coords)
        drmsd += calc_drmsd(predicted_coords, actual_coords)
    rmsd /= val_size
    drmsd /= val_size
    return val_size, rmsd, drmsd


def train_model(model):
    set_experiment_id("TRAIN")

    train_loader = contruct_dataloader_from_disk(TRAINING_FOLDER)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()

    best_model_loss = 1e20
    best_model_path = None
    stopping_condition_met = False
    minibatches_proccesed = 0

    train_loss_values = []
    validation_loss_values = []
    rmsd_avg_values = []
    drmsd_avg_values = []

    while not stopping_condition_met:
        optimizer.zero_grad()
        model.zero_grad()
        for minibatch_id, training_minibatch in enumerate(train_loader):
            # training_minibatch is a tuple of tuple of tensors except
            # lengths, which is a tuple of ints
            # Implies primary will be a tuple with MINIBATCH_SIZE number of elements
            # And each element will be of shape (Length,)
            # phi, psi and omega are in degrees here
            lengths, primary, evolutionary, actual_phi, actual_psi, actual_omega = (
                training_minibatch
            )
            # inp should be of shape [Batch, 41, Max_length]
            inp = model.generate_input(lengths, primary, evolutionary)
            # target should be of shape [Batch, 6, Max_length]
            target = model.generate_target(
                lengths, np.array([actual_phi, actual_psi, actual_omega])
            )
            # output should be of shape [Batch, 6, Max_length]
            # sin(phi), cos(phi), sin(psi), cos(psi), sin(omega), cos(omega)
            output = model(inp)
            loss = model.calculate_loss(lengths, criterion, output, target)
            optimizer.zero_grad()
            write_out("Train loss:", loss.item())
            loss.backward()
            optimizer.step()

            if minibatches_proccesed % EVAL_INTERVAL == 0:
                val_size, rmsd, drmsd = validate_model(model)
                if rmsd < best_model_loss:
                    best_model_loss = rmsd
                    best_model_path = write_model_to_disk(model)
                write_out("Validation loss: ", rmsd, "Train loss: ", loss.item())
                write_out("Best model stored at " + best_model_path)
                train_loss_values.append(loss)
                validation_loss_values.append(rmsd)
                rmsd_avg_values.append(rmsd)
                drmsd_avg_values.append(drmsd)
                if not HIDE_UI:
                    data = {}
                    data["pdb_data_pred"] = open("output/pred_test.pdb", "r").read()
                    data["pdb_data_true"] = open("output/actual_test.pdb", "r").read()
                    data["validation_dataset_size"] = val_size
                    # data["sample_num"] = sample_num
                    data["train_loss_values"] = train_loss_values
                    data["validation_loss_values"] = validation_loss_values
                    data["phi_actual"] = list(actual_phi)
                    data["psi_actual"] = list(actual_psi)
                    pred_phi = np.arctan2(output[:, 0, :], output[:, 1, :])
                    pred_psi = np.arctan2(output[:, 2, :], output[:, 3, :])
                    # pred_omega = np.arctan2(output[:, 4, :], output[:, 5, :])
                    data["phi_predicted"] = list(pred_phi * 180.0 / np.pi)
                    data["psi_predicted"] = list(pred_psi * 180.0 / np.pi)
                    data["drmsd_avg"] = drmsd_avg_values
                    data["rmsd_avg"] = rmsd_avg_values
                    res = requests.post("http://localhost:5000/graph", json=data)
                    if res.ok:
                        print(res.json())

            if minibatches_proccesed > MIN_BATCH_ITER:
                stopping_condition_met = True
                break
            minibatches_proccesed += 1

    write_result_summary(best_model_loss)
    return best_model_path
