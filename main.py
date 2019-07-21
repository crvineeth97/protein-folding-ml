# This file is part of the OpenProtein project.
# Modified for research usage
#
# @author Jeppe Hallgren
# @author Vineeth Chelur
# @author Yashas Samaga
#
# For license information, please see the LICENSE file in the root directory.

import math
import time

import numpy as np
import requests
import torch

from constants import EVAL_INTERVAL, LEARNING_RATE, MIN_UPDATES, HIDE_UI
from dashboard import start_dashboard_server
from dataloader import contruct_dataloader_from_disk
from models.resnet import ResNet
from preprocessing import process_raw_data
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


def train_model(data_set_identifier, train_folder, val_folder):
    set_experiment_id(data_set_identifier)

    train_loader = contruct_dataloader_from_disk(train_folder, device)
    validation_loader = contruct_dataloader_from_disk(val_folder, device)
    validation_dataset_size = validation_loader.dataset.__len__()

    model = ResNet(device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
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
        for minibatch_id, training_minibatch in enumerate(train_loader):
            minibatches_proccesed += 1
            # training_minibatch is a tuple of tuple of tensors except
            # lengths, which is a tuple of ints
            lengths, primary, evolutionary, phi, psi = training_minibatch

            start_compute_loss = time.time()

            # inp should be of shape [Batch, 41, Max_length]
            inp = model.generate_input(lengths, primary, evolutionary)  # .cuda()
            # target should be of shape [Batch, 4, Max_length]
            target = model.generate_target(lengths, phi, psi)
            # output should be of shape [Batch, 4, Max_length]
            # sin(phi), cos(phi), sin(psi), cos(psi)
            output = model(inp)
            loss = model.calculate_loss(lengths, criterion, output, target)
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
                angles = calculate_dihedral_angles(pos, device)
                angles_pred = calculate_dihedral_angles(pos_pred, device)
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
                if not HIDE_UI:
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


if HIDE_UI:
    write_out("Live plot deactivated, see output folder for plot.")

device = torch.device("cpu")
if torch.cuda.is_available():
    write_out("CUDA is available, using GPU")
    device = torch.device("cuda")

# Start web server
# TODO Add more options to view as well as use GDT_TS for scoring
if not HIDE_UI:
    start_dashboard_server()

start = time.time()
process_raw_data()
end = time.time()

print("Total Preprocessing Time: ", end - start)

training_folder = "data/preprocessed/training_30_no_missing/"
validation_folder = "data/preprocessed/validation_no_missing/"
testing_folder = "data/preprocessed/testing_no_missing/"

train_model_path = train_model("TRAIN", training_folder, validation_folder)
