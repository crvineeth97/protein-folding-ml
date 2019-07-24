import math
from datetime import datetime

import numpy as np
import torch
from Bio.PDB import PDBIO
from PeptideBuilder import make_structure

from constants import LEARNING_RATE, MINIBATCH_SIZE


def set_experiment_id(data_set_identifier):
    output_string = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    output_string += "-" + data_set_identifier
    output_string += "-LR" + str(LEARNING_RATE).replace(".", "_")
    output_string += "-MB" + str(MINIBATCH_SIZE)
    globals().__setitem__("experiment_id", output_string)


def write_out(*args, end="\n"):
    output_string = (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        + ": "
        + str.join(" ", [str(a) for a in args])
        + end
    )
    if globals().get("experiment_id") is not None:
        with open(
            "output/" + globals().get("experiment_id") + ".txt", "a+"
        ) as output_file:
            output_file.write(output_string)
            output_file.flush()
    print(output_string, end="")


def write_model_to_disk(model):
    path = "output/models/" + globals().get("experiment_id") + ".model"
    torch.save(model, path)
    return path


def write_to_pdb(primary, dihedrals, name):
    # dihedrals are in degrees
    phi, psi, omega = dihedrals
    structure = make_structure(primary, phi[1:], psi[1:], omega[1:])
    out = PDBIO()
    out.set_structure(structure)
    out.save("output/" + name + ".pdb")


def draw_plot(
    fig,
    plt,
    validation_dataset_size,
    sample_num,
    train_loss_values,
    validation_loss_values,
):
    def draw_with_vars():
        ax = fig.gca()
        ax2 = ax.twinx()
        plt.grid(True)
        plt.title(
            "Training progress ("
            + str(validation_dataset_size)
            + " samples in validation set)"
        )
        train_loss_plot, = ax.plot(sample_num, train_loss_values)
        ax.set_ylabel("Train Negative log likelihood")
        ax.yaxis.labelpad = 0
        validation_loss_plot, = ax2.plot(
            sample_num, validation_loss_values, color="black"
        )
        ax2.set_ylabel("Validation loss")
        ax2.set_ylim(bottom=0)
        plt.legend(
            [train_loss_plot, validation_loss_plot],
            ["Train loss on last batch", "Validation loss"],
        )
        ax.set_xlabel("Minibatches processed (=network updates)", color="black")

    return draw_with_vars


def draw_ramachandran_plot(fig, plt, phi, psi):
    def draw_with_vars():
        ax = fig.gca()
        plt.grid(True)
        plt.title("Ramachandran plot")
        train_loss_plot, = ax.plot(phi, psi)
        ax.set_ylabel("Psi")
        ax.yaxis.labelpad = 0
        plt.legend([train_loss_plot], ["Phi psi"])
        ax.set_xlabel("Phi", color="black")

    return draw_with_vars


def write_result_summary(accuracy):
    output_string = globals().get("experiment_id") + ": " + str(accuracy) + "\n"
    with open("output/result_summary.txt", "a+") as output_file:
        output_file.write(output_string)
        output_file.flush()
    print(output_string, end="")


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
