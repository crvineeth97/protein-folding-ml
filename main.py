# This file is part of the OpenProtein project.
# Modified for research usage
#
# @author Jeppe Hallgren
# @author Vineeth Chelur
# @author Yashas Samaga
#
# For license information, please see the LICENSE file in the root directory.

import argparse
import math
import time

import numpy as np
import requests
import torch
import torch.optim as optim
import torch.utils.data

from dashboard import start_dashboard_server
from models.angle_pred import NETWORK_INPUT_RESIDUES_MAX
from models.angle_pred import Network as ResNetModel
from models.lstm import LSTMModel
from parameters import EVAL_INTERVAL, LEARNING_RATE, MIN_UPDATES, MINIBATCH_SIZE
from preprocessing import process_raw_data
from util import (
    calculate_dihedral_angels,
    contruct_dataloader_from_disk,
    evaluate_model,
    get_structure_from_angles,
    set_experiment_id,
    write_model_to_disk,
    write_out,
    write_result_summary,
    write_to_pdb,
)

# TODO Add more arguments
parser = argparse.ArgumentParser(description="OpenProtein version 0.1")
parser.add_argument(
    "--hide-ui",
    dest="hide_ui",
    action="store_true",
    default=False,
    help="Hide loss graph and visualization UI while training goes on.",
)
parser.add_argument(
    "--evaluate-on-test",
    dest="evaluate_on_test",
    action="store_true",
    default=False,
    help="Run model on test data.",
)
parser.add_argument(
    "--log-file",
    dest="log_file",
    type=str,
    default="./output/train.log",
    help="Print output to the given log file",
)
parser.add_argument(
    "--debug-mode",
    dest="debug_mode",
    action="store_true",
    default=False,
    help="Enter debugging mode for a more detailed log",
)
parser.add_argument(
    "--force-pre-processing-overwrite",
    dest="force_pre_processing_overwrite",
    action="store_true",
    default=False,
    help="Deletes already preprocessed data in data/preprocessed and uses the raw data again",
)


def train_model(data_set_identifier, train_file, val_file):
    set_experiment_id(data_set_identifier, LEARNING_RATE, MINIBATCH_SIZE)

    train_loader = contruct_dataloader_from_disk(train_file, MINIBATCH_SIZE, device)
    validation_loader = contruct_dataloader_from_disk(val_file, MINIBATCH_SIZE, device)
    validation_dataset_size = validation_loader.dataset.__len__()

    model = LSTMModel(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()

    sample_num = list()
    train_loss_values = list()
    validation_loss_values = list()
    rmsd_avg_values = list()
    drmsd_avg_values = list()

    best_model_loss = 1e20
    best_model_minibatch_time = None
    best_model_path = None
    stopping_condition_met = False
    minibatches_proccesed = 0

    while not stopping_condition_met:
        loss_tracker = np.zeros(0)
        for minibatch_id, training_minibatch in enumerate(train_loader, 0):
            minibatches_proccesed += 1
            lengths, primary, evolutionary, phi, psi = training_minibatch
            print(primary)

            start_compute_loss = time.time()
            # inp should be the feature vectors to send to the particular model
            # primary is of shape [MINIBATCH_SIZE, MAX_SEQ_LEN]
            inp = model.generate_input(primary, evolutionary, lengths)
            # output of the model
            # In our case: sin(phi), cos(phi), sin(psi), cos(psi)
            output = model(inp)
            target = torch.tensor(
                [torch.sin(phi), torch.cos(phi), torch.sin(psi), torch.cos(psi)],
                device=device,
            )
            loss = criterion(output, target)
            optimizer.zero_grad()
            write_out("Train loss:", loss.item())
            start_compute_grad = time.time()
            loss.backward()
            loss_tracker = np.append(loss_tracker, loss.item())
            end = time.time()
            write_out(
                "Loss time: ",
                start_compute_grad - start_compute_loss,
                "Grad time: ",
                end - start_compute_grad,
            )
            optimizer.step()

            # for every EVAL_INTERVAL samples,
            # plot performance on the validation set
            if minibatches_proccesed % EVAL_INTERVAL == 0:

                write_out("Testing model on validation set...")

                train_loss = loss_tracker.mean()
                loss_tracker = np.zeros(0)
                validation_loss, data_total, rmsd_avg, drmsd_avg = evaluate_model(
                    validation_loader, model
                )
                prim = data_total[0][0]
                pos = data_total[0][1]
                pos_pred = data_total[0][3]
                angles = calculate_dihedral_angels(pos, device)
                angles_pred = calculate_dihedral_angels(pos_pred, device)
                write_to_pdb(get_structure_from_angles(prim, angles), "test")
                write_to_pdb(get_structure_from_angles(prim, angles_pred), "test_pred")
                if validation_loss < best_model_loss:
                    best_model_loss = validation_loss
                    best_model_minibatch_time = minibatches_proccesed
                    best_model_path = write_model_to_disk(model)

                write_out(
                    "Validation loss:", validation_loss, "Train loss:", train_loss
                )
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
                if not ARGS.hide_ui:
                    data = {}
                    data["pdb_data_pred"] = open(
                        "output/protein_test_pred.pdb", "r"
                    ).read()
                    data["pdb_data_true"] = open("output/protein_test.pdb", "r").read()
                    data["validation_dataset_size"] = validation_dataset_size
                    data["sample_num"] = sample_num
                    data["train_loss_values"] = train_loss_values
                    data["validation_loss_values"] = validation_loss_values
                    data["phi_actual"] = list(
                        [math.degrees(float(v)) for v in angles[1:, 1]]
                    )
                    data["psi_actual"] = list(
                        [math.degrees(float(v)) for v in angles[:-1, 2]]
                    )
                    data["phi_predicted"] = list(
                        [math.degrees(float(v)) for v in angles_pred[1:, 1]]
                    )
                    data["psi_predicted"] = list(
                        [math.degrees(float(v)) for v in angles_pred[:-1, 2]]
                    )
                    data["drmsd_avg"] = drmsd_avg_values
                    data["rmsd_avg"] = rmsd_avg_values
                    res = requests.post("http://localhost:5000/graph", json=data)
                    if res.ok:
                        print(res.json())

                if (
                    minibatches_proccesed > MIN_UPDATES
                    and minibatches_proccesed > best_model_minibatch_time * 2
                ):
                    stopping_condition_met = True
                    break
    write_result_summary(best_model_loss)
    return best_model_path


def train_model_resnet(data_set_identifier, train_file, val_file):
    set_experiment_id(data_set_identifier, LEARNING_RATE, MINIBATCH_SIZE)

    train_loader = contruct_dataloader_from_disk(train_file, MINIBATCH_SIZE, None)
    validation_loader = contruct_dataloader_from_disk(val_file, MINIBATCH_SIZE, None)
    validation_dataset_size = len(validation_loader.dataset)

    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    model = ResNetModel(device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()

    sample_num = list()
    train_loss_values = list()
    validation_loss_values = list()
    rmsd_avg_values = list()
    drmsd_avg_values = list()

    best_model_loss = 1e20
    best_model_minibatch_time = None
    best_model_path = None
    stopping_condition_met = False
    minibatches_proccesed = 0

    while not stopping_condition_met:
        loss_tracker = np.zeros(0)
        for minibatch_id, training_minibatch in enumerate(train_loader, 0):
            minibatches_proccesed += 1
            lengths, primary, evolutionary, phi, psi = training_minibatch

            batch_size = len(primary)
            transformed_phi = torch.zeros(batch_size, 1, NETWORK_INPUT_RESIDUES_MAX)
            transformed_psi = torch.zeros(batch_size, 1, NETWORK_INPUT_RESIDUES_MAX)

            for i in range(batch_size):
                transformed_phi[i] = phi[i]
                transformed_psi[i] = psi[i]

            start_compute_loss = time.time()

            # expected input shape: [n, 40, L]
            inp = model.generate_input(evolutionary, primary, lengths)  # .cuda()

            # expected output shape: [n, 4, L]
            # sin(phi), cos(phi), sin(psi), cos(psi)
            output = model(inp)
            sin_phi_tensor = torch.sin(transformed_phi)
            cos_phi_tensor = torch.cos(transformed_phi)
            sin_psi_tensor = torch.sin(transformed_psi)
            cos_psi_tensor = torch.cos(transformed_psi)
            target = torch.stack(
                (sin_phi_tensor, cos_phi_tensor, sin_psi_tensor, cos_psi_tensor),
                device=device,
            )
            target = target.permute(2, 1, 0, 3)
            target = target.reshape(1, batch_size, -1)
            target = target.squeeze()
            loss = criterion(output, target)
            optimizer.zero_grad()
            write_out("Train loss:", loss.item())
            start_compute_grad = time.time()
            loss.backward()
            loss_tracker = np.append(loss_tracker, loss.item())
            end = time.time()
            write_out(
                "Loss time: ",
                start_compute_grad - start_compute_loss,
                "Grad time: ",
                end - start_compute_grad,
            )
            optimizer.step()

            # for every EVAL_INTERVAL samples,
            # plot performance on the validation set
            if minibatches_proccesed % EVAL_INTERVAL == 0:

                write_out("Testing model on validation set...")

                train_loss = loss_tracker.mean()
                loss_tracker = np.zeros(0)
                validation_loss, data_total, rmsd_avg, drmsd_avg = evaluate_model(
                    validation_loader, model
                )
                prim = data_total[0][0]
                pos = data_total[0][1]
                pos_pred = data_total[0][3]
                angles = calculate_dihedral_angels(pos, device)
                angles_pred = calculate_dihedral_angels(pos_pred, device)
                write_to_pdb(get_structure_from_angles(prim, angles), "test")
                write_to_pdb(get_structure_from_angles(prim, angles_pred), "test_pred")
                if validation_loss < best_model_loss:
                    best_model_loss = validation_loss
                    best_model_minibatch_time = minibatches_proccesed
                    best_model_path = write_model_to_disk(model)

                write_out(
                    "Validation loss:", validation_loss, "Train loss:", train_loss
                )
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
                if not ARGS.hide_ui:
                    data = {}
                    data["pdb_data_pred"] = open(
                        "output/protein_test_pred.pdb", "r"
                    ).read()
                    data["pdb_data_true"] = open("output/protein_test.pdb", "r").read()
                    data["validation_dataset_size"] = validation_dataset_size
                    data["sample_num"] = sample_num
                    data["train_loss_values"] = train_loss_values
                    data["validation_loss_values"] = validation_loss_values
                    data["phi_actual"] = list(
                        [math.degrees(float(v)) for v in angles[1:, 1]]
                    )
                    data["psi_actual"] = list(
                        [math.degrees(float(v)) for v in angles[:-1, 2]]
                    )
                    data["phi_predicted"] = list(
                        [math.degrees(float(v)) for v in angles_pred[1:, 1]]
                    )
                    data["psi_predicted"] = list(
                        [math.degrees(float(v)) for v in angles_pred[:-1, 2]]
                    )
                    data["drmsd_avg"] = drmsd_avg_values
                    data["rmsd_avg"] = rmsd_avg_values
                    res = requests.post("http://localhost:5000/graph", json=data)
                    if res.ok:
                        print(res.json())

                if (
                    minibatches_proccesed > ARGS.minimum_updates
                    and minibatches_proccesed > best_model_minibatch_time * 2
                ):
                    stopping_condition_met = True
                    break
    write_result_summary(best_model_loss)
    return best_model_path


ARGS = parser.parse_known_args()[0]
device = torch.device("cpu")

if ARGS.hide_ui:
    write_out("Live plot deactivated, see output folder for plot.")

if torch.cuda.is_available():
    write_out("CUDA is available, using GPU")
    device = torch.device("cuda")

# Start web server
# TODO Add more options to view as well as use GDT_TS for scoring
if not ARGS.hide_ui:
    start_dashboard_server()

start = time.time()
process_raw_data(force_pre_processing_overwrite=False)
end = time.time()

print("Total Preprocessing Time: ", end - start)

training_file = "data/preprocessed/training_30.hdf5"
validation_file = "data/preprocessed/validation.hdf5"
testing_file = "data/preprocessed/testing.hdf5"

train_model_path = train_model_resnet("TRAIN", training_file, validation_file)

# print(train_model_path)
