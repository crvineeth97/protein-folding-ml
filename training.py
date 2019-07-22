import numpy as np
from constants import EVAL_INTERVAL, LEARNING_RATE, MIN_UPDATES
from dataloader import contruct_dataloader_from_disk
from visualization import visualize
import torch
from util import (
    calculate_dihedral_angles,
    evaluate_model,
    get_structure_from_angles,
    set_experiment_id,
    write_model_to_disk,
    write_out,
    write_result_summary,
    write_to_pdb,
)
import pnerf.pnerf as pnerf

TRAINING_FOLDER = "data/preprocessed/training_30_no_missing/"
VALIDATION_FOLDER = "data/preprocessed/validation_no_missing/"


def validate_model(model, train_loss, device):
    validation_loader = contruct_dataloader_from_disk(VALIDATION_FOLDER)
    loss = 0
    dRMSD_list = []
    RMSD_list = []
    for i, data in enumerate(validation_loader):
        lengths, primary, evolutionary, phi, psi, omega, tertiary = data
        inp = model.generate_input(lengths, primary, evolutionary)
        output = model(inp)
        phi = np.arctan2(output[:, 0, :], output[:, 1, :])
        psi = np.arctan2(output[:, 2, :], output[:, 3, :])
        omega = np.arctan2(output[:, 4, :], output[:, 5, :])

        dihedrals = torch.from_numpy(
            np.reshape(
                np.array(list(zip(phi, psi, omega)), dtype=np.float32), (-1, 1, 3)
            )
        )
        points = pnerf.dihedral_to_point(dihedrals, device)
        coordinates = (
            pnerf.point_to_coordinate(points, device) / 100
        )  # divide by 100 to get in angstrom units
        coords = coordinates.transpose(0, 1).contiguous().view(1, -1, 9).transpose(0, 1)
        predicted_coords = coords.transpose(0, 1).contiguous().view(-1, 3)
        actual_coords = (
            torch.from_numpy(tertiary)
            .unsqueeze(1)
            .transpose(0, 1)
            .contiguous()
            .view(-1, 3)
        )
        rmsd = calc_rmsd(predicted_coords, actual_coords)
        drmsd = calc_drmsd(predicted_coords, actual_coords)

        write_out("Apply model to validation minibatch:", time.time() - start)
        cpu_predicted_angles = predicted_angles.transpose(0, 1).cpu().detach()
        cpu_predicted_backbone_atoms = backbone_atoms.transpose(0, 1).cpu().detach()
        minibatch_data = list(
            zip(primary, tertiary, cpu_predicted_angles, cpu_predicted_backbone_atoms)
        )
        data_total.extend(minibatch_data)
        start = time.time()
        for (
            primary_sequence,
            tertiary_positions,
            predicted_pos,
            predicted_backbone_atoms,
        ) in minibatch_data:
            actual_coords = tertiary_positions.transpose(0, 1).contiguous().view(-1, 3)
            predicted_coords = (
                predicted_backbone_atoms[: len(primary_sequence)]
                .transpose(0, 1)
                .contiguous()
                .view(-1, 3)
                .detach()
            )
            rmsd = calc_rmsd(predicted_coords, actual_coords)
            drmsd = calc_drmsd(predicted_coords, actual_coords)
            RMSD_list.append(rmsd)
            dRMSD_list.append(drmsd)
            error = rmsd
            loss += error
            end = time.time()
        write_out("Calculate validation loss for minibatch took:", end - start)
    loss /= data_loader.dataset.__len__()

    write_to_pdb(get_structure_from_angles(prim, angles), "test")
    write_to_pdb(get_structure_from_angles(prim, angles_pred), "test_pred")
    if validation_loss < best_model_loss:
        best_model_loss = validation_loss
        best_model_minibatch_time = minibatches_proccesed
        best_model_path = write_model_to_disk(model)

    write_out("Validation loss:", validation_loss, "Train loss:", train_loss)
    write_out(
        "Best model so far (validation loss): ",
        validation_loss,
        "at time",
        best_model_minibatch_time,
    )
    write_out("Best model stored at " + best_model_path)
    write_out("Minibatches processed:", minibatches_proccesed)
    sample_num.append(minibatches_proccesed)
    train_loss_values.append(train_loss)
    validation_loss_values.append(validation_loss)
    rmsd_avg_values.append(rmsd_avg)
    drmsd_avg_values.append(drmsd_avg)
    visualize()


def train_model(model, device):
    set_experiment_id("TRAIN")

    train_loader = contruct_dataloader_from_disk(TRAINING_FOLDER)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()

    best_model_loss = 1e20
    best_model_path = None
    stopping_condition_met = False
    minibatches_proccesed = 0

    while not stopping_condition_met:
        optimizer.zero_grad()
        model.zero_grad()
        loss_tracker = np.zeros(0)
        for minibatch_id, training_minibatch in enumerate(train_loader):
            minibatches_proccesed += 1
            # training_minibatch is a tuple of tuple of tensors except
            # lengths, which is a tuple of ints
            lengths, primary, evolutionary, phi, psi, omega = training_minibatch
            # inp should be of shape [Batch, 41, Max_length]
            inp = model.generate_input(lengths, primary, evolutionary)  # .cuda()
            # target should be of shape [Batch, 6, Max_length]
            target = model.generate_target(lengths, [phi, psi, omega])
            # output should be of shape [Batch, 6, Max_length]
            # sin(phi), cos(phi), sin(psi), cos(psi), sin(omega), cos(omega)
            output = model(inp)
            loss = model.calculate_loss(lengths, criterion, output, target)
            optimizer.zero_grad()
            write_out("Train loss:", loss.item())
            loss.backward()
            loss_tracker = np.append(loss_tracker, loss.item())
            optimizer.step()

            if minibatches_proccesed % EVAL_INTERVAL == 0:
                validate_model(model, loss_tracker.mean())
                loss_tracker = np.zeros(0)

            if minibatches_proccesed > MIN_UPDATES:
                stopping_condition_met = True
                break

    write_result_summary(best_model_loss)
    return best_model_path
