import logging
import math
import os
import sys
from datetime import datetime

import numpy as np
import torch

from constants import (
    DEVICE,
    EVAL_INTERVAL,
    LEARNING_RATE,
    MINIBATCH_SIZE,
    PREPROCESS_PROTEIN_WITH_MISSING_RESIDUES,
    PRINT_LOSS_INTERVAL,
    TRAINING_EPOCHS,
)


def init_output_dir(model):
    model_dir = datetime.now().strftime("%Y-%m-%d %H_%M_%S")
    globals().__setitem__("experiment_id", model_dir)
    model_dir = "output/" + model_dir + "/"
    os.makedirs(model_dir)
    os.makedirs(model_dir + "code/")
    # Keep a copy of .sh and .py files in the model's code
    os.system(
        "rsync -mar --include='*/' --include='*\.py' "
        + "--include='*\.sh' --exclude='*' --exclude='./output/*' ./ "
        + "\ ".join(model_dir.split(" "))
        + "code/"
    )
    file_handler = logging.FileHandler(filename=model_dir + "log.txt")
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
        handlers=handlers,
    )
    logging.info("DEVICE: %s", DEVICE)
    logging.info("EVAL_INTERVAL: %d", EVAL_INTERVAL)
    logging.info("LEARNING_RATE: %f", LEARNING_RATE)
    logging.info("MINIBATCH_SIZE: %d", MINIBATCH_SIZE)
    logging.info(
        "PREPROCESS_PROTEIN_WITH_MISSING_RESIDUES: %s",
        str(PREPROCESS_PROTEIN_WITH_MISSING_RESIDUES),
    )
    logging.info("PRINT_LOSS_INTERVAL: %d", PRINT_LOSS_INTERVAL)
    logging.info("TRAINING_EPOCHS: %d", TRAINING_EPOCHS)
    logging.info(model)


def get_model_dir():
    model_dir = globals().__getitem__("experiment_id")
    model_dir = "output/" + model_dir + "/"
    return model_dir


def write_model_to_disk(model, name):
    path = "output/" + globals().get("experiment_id") + "/" + name + ".model"
    torch.save(model, path)


def calc_pairwise_distances(chain_a, chain_b, device):
    distance_matrix = torch.Tensor(
        chain_a.size()[0], chain_b.size()[0], device=device
    ).type(torch.float)
    # add small epsilon to avoid boundary issues
    epsilon = 10 ** (-4) * torch.ones(chain_a.size(0), chain_b.size(0), device=device)
    for i, row in enumerate(chain_a.split(1)):
        distance_matrix[i] = torch.sum((row.expand_as(chain_b) - chain_b) ** 2, 1).view(
            1, -1
        )

    return torch.sqrt(distance_matrix + epsilon)


# method for translating a point cloud to its center of mass
def transpose_atoms_to_center_of_mass(x):
    # calculate com by summing x, y and z respectively
    # and dividing by the number of points
    centerOfMass = np.array(
        [
            [x[0, :].sum() / x.shape[1]],
            [x[1, :].sum() / x.shape[1]],
            [x[2, :].sum() / x.shape[1]],
        ]
    )
    # translate points to com and return
    return x - centerOfMass


def calc_rmsd(chain_a, chain_b):
    # move to center of mass
    a = chain_a.cpu().numpy().transpose()
    b = chain_b.cpu().numpy().transpose()
    X = transpose_atoms_to_center_of_mass(a)
    Y = transpose_atoms_to_center_of_mass(b)

    R = np.matmul(Y, X.transpose())
    # extract the singular values
    _, S, _ = np.linalg.svd(R)
    # compute RMSD using the formula
    E0 = sum(
        list(np.linalg.norm(x) ** 2 for x in X.transpose())
        + list(np.linalg.norm(x) ** 2 for x in Y.transpose())
    )
    TraceS = sum(S)
    RMSD = np.sqrt((1 / len(X.transpose())) * (E0 - 2 * TraceS))
    return RMSD


def calc_drmsd(chain_a, chain_b, device=torch.device("cpu")):
    assert len(chain_a) == len(chain_b)
    distance_matrix_a = calc_pairwise_distances(chain_a, chain_a, device)
    distance_matrix_b = calc_pairwise_distances(chain_b, chain_b, device)
    return torch.norm(distance_matrix_a - distance_matrix_b, 2) / math.sqrt(
        (len(chain_a) * (len(chain_a) - 1))
    )


def calc_angular_difference(a1, a2):
    a1 = a1.transpose(0, 1).contiguous()
    a2 = a2.transpose(0, 1).contiguous()
    sum = 0
    for idx, _ in enumerate(a1):
        assert a1[idx].shape[1] == 3
        assert a2[idx].shape[1] == 3
        a1_element = a1[idx].view(-1, 1)
        a2_element = a2[idx].view(-1, 1)
        sum += torch.sqrt(
            torch.mean(
                torch.min(
                    torch.abs(a2_element - a1_element),
                    2 * math.pi - torch.abs(a2_element - a1_element),
                )
                ** 2
            )
        )
    return sum / a1.shape[0]


def structure_to_backbone_atoms(structure):
    predicted_coords = []
    for res in structure.get_residues():
        predicted_coords.append(torch.Tensor(res["N"].get_coord()))
        predicted_coords.append(torch.Tensor(res["CA"].get_coord()))
        predicted_coords.append(torch.Tensor(res["C"].get_coord()))
    return torch.stack(predicted_coords).view(-1, 9)


def calc_avg_drmsd_over_minibatch(
    backbone_atoms_padded, actual_coords_padded, batch_sizes
):
    backbone_atoms_list = list(
        [
            backbone_atoms_padded[: batch_sizes[i], i]
            for i in range(int(backbone_atoms_padded.size(1)))
        ]
    )
    actual_coords_list = list(
        [
            actual_coords_padded[: batch_sizes[i], i]
            for i in range(int(actual_coords_padded.size(1)))
        ]
    )
    drmsd_avg = 0
    for idx, backbone_atoms in enumerate(backbone_atoms_list):
        actual_coords = actual_coords_list[idx].transpose(0, 1).contiguous().view(-1, 3)
        drmsd_avg += calc_drmsd(
            backbone_atoms.transpose(0, 1).contiguous().view(-1, 3), actual_coords
        )
    return drmsd_avg / len(backbone_atoms_list)
