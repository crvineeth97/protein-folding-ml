import logging
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
        "rsync -mar --exclude='output' --include='*/' "
        + "--include='*\.py' --include='*\.sh' --exclude='*' ./ "
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


def generate_input(lengths, primary, evolutionary):
    """
    Generate input for each minibatch. Pad the input feature vectors
    so that the final input shape is [MINIBATCH_SIZE, 41, Max_length]
    Args:
    lengths: Tuple of all protein lengths in current minibatch
    primary: Tuple of numpy arrays of shape (l,) describing the
    protein amino acid sequence, which are of variable length
    evolutionary: Tuple of numpy arrays of shape (l, 21) describing
    the PSSM matrix of the protein
    """
    batch_size = len(lengths)
    transformed_primary = torch.zeros(
        batch_size, 20, lengths[0], device=DEVICE, dtype=torch.float32
    )

    # TODO: Use pythonic way
    for i in range(batch_size):
        for j in range(lengths[i]):
            residue = int(primary[i][j])
            transformed_primary[i][residue][j] = 1.0

    transformed_evolutionary = torch.zeros(
        batch_size, 21, lengths[0], device=DEVICE, dtype=torch.float32
    )
    for i in range(batch_size):
        transformed_evolutionary[i, :, : lengths[i]] = torch.from_numpy(
            evolutionary[i].T
        )
    # transformed_primary           [n, 20, L]
    # transformed_evolutionary      [n, 21, L]
    # output                        [n, 41, L]
    return torch.cat((transformed_primary, transformed_evolutionary), dim=1)


def generate_target(lengths, phi, psi, omega):
    # dihedrals are in radians
    batch_size = len(lengths)
    target = torch.zeros(batch_size, 4, lengths[0], device=DEVICE, dtype=torch.float32)
    for i in range(batch_size):
        ph = torch.from_numpy(phi[i])
        ps = torch.from_numpy(psi[i])
        # om = torch.from_numpy(omega[i])
        target[i, 0, : lengths[i]] = torch.sin(ph)
        target[i, 1, : lengths[i]] = torch.cos(ph)
        target[i, 2, : lengths[i]] = torch.sin(ps)
        target[i, 3, : lengths[i]] = torch.cos(ps)
        # target[i, 4, : lengths[i]] = torch.sin(om)
        # target[i, 5, : lengths[i]] = torch.cos(om)
    return target


def calculate_loss(lengths, criterion, output, target):
    batch_size = len(lengths)
    loss = criterion(output[0], target[0])
    for i in range(1, batch_size):
        loss += criterion(output[i, :, : lengths[i]], target[i, :, : lengths[i]])
    loss /= batch_size
    return loss


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


# TODO Fix calculation
def calculate_rmsd(chain_a, chain_b, lengths):
    chain_a = chain_a.cpu().numpy()
    chain_b = chain_b.cpu().numpy()
    RMSD = 0
    bs = len(lengths)
    for i in range(bs):
        a = chain_a[i, : lengths[i]].transpose()
        b = chain_b[i, : lengths[i]].transpose()
        # move to center of mass
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
        RMSD += np.sqrt((1 / len(X.transpose())) * (E0 - 2 * TraceS))
    RMSD /= bs
    return RMSD
