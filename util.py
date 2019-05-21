# This file is part of the OpenProtein project.
#
# @author Jeppe Hallgren
#
# For license information, please see the LICENSE file in the root directory.

import math
import time
from datetime import datetime
import warnings

import h5py
import numpy as np
import torch
import torch.utils.data
from Bio.PDB import PDBIO
from PeptideBuilder import make_structure

import pnerf.pnerf as pnerf

warnings.filterwarnings("error")
AA_ID_DICT = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
}

DSSP_DICT = {"L": 0, "H": 1, "B": 2, "E": 3, "G": 4, "I": 5, "T": 6, "S": 7}
MASK_DICT = {"-": 0, "+": 1}


def contruct_dataloader_from_disk(filename, minibatch_size, device):
    return torch.utils.data.DataLoader(
        H5PytorchDataset(filename, device),
        batch_size=minibatch_size,
        shuffle=True,
        collate_fn=merge_samples_to_minibatch,
    )


class H5PytorchDataset(torch.utils.data.Dataset):
    def __init__(self, filename, device):
        super(H5PytorchDataset, self).__init__()

        self.h5pyfile = h5py.File(filename, "r")
        self.device = device
        self.num_proteins, self.max_sequence_len = self.h5pyfile["primary"].shape

    def __getitem__(self, index):
        lengths = torch.tensor(
            self.h5pyfile["length"][index, :], dtype=torch.int32, device=self.device
        )
        primary = torch.tensor(
            self.h5pyfile["primary"][index, :], dtype=torch.uint8, device=self.device
        )
        evolutionary = torch.tensor(
            self.h5pyfile["evolutionary"][index, :],
            dtype=torch.float,
            device=self.device,
        )
        # secondary = torch.tensor(
        #     self.h5pyfile["secondary"][index, :], dtype=torch.uint8, device=self.device
        # )
        phi = torch.tensor(
            self.h5pyfile["phi"][index, :], dtype=torch.float, device=self.device
        )
        psi = torch.tensor(
            self.h5pyfile["psi"][index, :], dtype=torch.float, device=self.device
        )
        return lengths, primary, evolutionary, phi, psi

    def __len__(self):
        return self.num_proteins


def merge_samples_to_minibatch(samples):
    samples_list = []
    for s in samples:
        samples_list.append(s)
    # sort according to length of aa sequence
    samples_list.sort(key=lambda x: len(x[0]), reverse=True)
    return zip(*samples_list)


def set_experiment_id(data_set_identifier, learning_rate, minibatch_size):
    output_string = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    output_string += "-" + data_set_identifier
    output_string += "-LR" + str(learning_rate).replace(".", "_")
    output_string += "-MB" + str(minibatch_size)
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


def evaluate_model(data_loader, model):
    loss = 0
    data_total = []
    dRMSD_list = []
    RMSD_list = []
    for i, data in enumerate(data_loader, 0):
        primary_sequence, tertiary_positions, mask = data
        start = time.time()
        predicted_angles, backbone_atoms, batch_sizes = model(primary_sequence)
        write_out("Apply model to validation minibatch:", time.time() - start)
        cpu_predicted_angles = predicted_angles.transpose(0, 1).cpu().detach()
        cpu_predicted_backbone_atoms = backbone_atoms.transpose(0, 1).cpu().detach()
        minibatch_data = list(
            zip(
                primary_sequence,
                tertiary_positions,
                cpu_predicted_angles,
                cpu_predicted_backbone_atoms,
            )
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
    return (
        loss,
        data_total,
        float(torch.Tensor(RMSD_list).mean()),
        float(torch.Tensor(dRMSD_list).mean()),
    )


def write_model_to_disk(model):
    path = "output/models/" + globals().get("experiment_id") + ".model"
    torch.save(model, path)
    return path


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


def calculate_dihedral_angles_over_minibatch(atomic_coords_padded, batch_sizes, device):
    angles = []
    atomic_coords = atomic_coords_padded.transpose(0, 1)
    for idx, _ in enumerate(batch_sizes):
        angles.append(
            calculate_dihedral_angels(atomic_coords[idx][: batch_sizes[idx]], device)
        )
    return torch.nn.utils.rnn.pad_packed_sequence(
        torch.nn.utils.rnn.pack_sequence(angles)
    )


def protein_id_to_str(protein_id_list):
    _aa_dict_inverse = {v: k for k, v in AA_ID_DICT.items()}
    aa_list = []
    for a in protein_id_list:
        aa_symbol = _aa_dict_inverse[int(a)]
        aa_list.append(aa_symbol)
    return aa_list


def calculate_dihedral_angels(atomic_coords, device):

    assert int(atomic_coords.shape[1]) == 9
    atomic_coords = atomic_coords.contiguous().view(-1, 3)

    zero_tensor = torch.tensor(0.0, device=device)

    dihedral_list = [zero_tensor, zero_tensor]
    dihedral_list.extend(compute_dihedral_list(atomic_coords))
    dihedral_list.append(zero_tensor)
    angles = torch.tensor(dihedral_list).view(-1, 3)
    return angles


def compute_dihedral_list(atomic_coords):
    # atomic_coords is -1 x 3
    ba = atomic_coords[1:] - atomic_coords[:-1]
    ba /= ba.norm(dim=1).unsqueeze(1)
    ba_neg = -1 * ba

    n1_vec = torch.cross(ba[:-2], ba_neg[1:-1], dim=1)
    n2_vec = torch.cross(ba_neg[1:-1], ba[2:], dim=1)
    n1_vec /= n1_vec.norm(dim=1).unsqueeze(1)
    n2_vec /= n2_vec.norm(dim=1).unsqueeze(1)

    m1_vec = torch.cross(n1_vec, ba_neg[1:-1], dim=1)

    x = torch.sum(n1_vec * n2_vec, dim=1)
    y = torch.sum(m1_vec * n2_vec, dim=1)

    return torch.atan2(y, x)


def get_structure_from_angles(aa_list_encoded, angles):
    aa_list = protein_id_to_str(aa_list_encoded)
    omega_list = angles[1:, 0]
    phi_list = angles[1:, 1]
    psi_list = angles[:-1, 2]
    assert len(aa_list) == len(phi_list) + 1 == len(psi_list) + 1 == len(omega_list) + 1
    structure = make_structure(
        aa_list,
        list(map(lambda x: math.degrees(x), phi_list)),
        list(map(lambda x: math.degrees(x), psi_list)),
        list(map(lambda x: math.degrees(x), omega_list)),
    )
    return structure


def write_to_pdb(structure, prot_id):
    out = PDBIO()
    out.set_structure(structure)
    out.save("output/protein_" + str(prot_id) + ".pdb")


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


def calc_drmsd(chain_a, chain_b, device=torch.device("cpu")):
    assert len(chain_a) == len(chain_b)
    distance_matrix_a = calc_pairwise_distances(chain_a, chain_a, device)
    distance_matrix_b = calc_pairwise_distances(chain_b, chain_b, device)
    return torch.norm(distance_matrix_a - distance_matrix_b, 2) / math.sqrt(
        (len(chain_a) * (len(chain_a) - 1))
    )


# method for translating a point cloud to its center of mass


def transpose_atoms_to_center_of_mass(x):
    # calculate com by summing x, y and z respectively
    # and dividing by the number of points
    centerOfMass = np.matrix(
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

    R = Y * X.transpose()
    # extract the singular values
    _, S, _ = np.linalg.svd(R)
    # compute RMSD using the formular
    E0 = sum(
        list(np.linalg.norm(x) ** 2 for x in X.transpose())
        + list(np.linalg.norm(x) ** 2 for x in Y.transpose())
    )
    TraceS = sum(S)
    RMSD = np.sqrt((1 / len(X.transpose())) * (E0 - 2 * TraceS))
    return RMSD


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


def structures_to_backbone_atoms_padded(structures):
    backbone_atoms_list = []
    for structure in structures:
        backbone_atoms_list.append(structure_to_backbone_atoms(structure))
    backbone_atoms_padded, batch_sizes_backbone = torch.nn.utils.rnn.pad_packed_sequence(
        torch.nn.utils.rnn.pack_sequence(backbone_atoms_list)
    )
    return backbone_atoms_padded, batch_sizes_backbone


def structure_to_backbone_atoms(structure):
    predicted_coords = []
    for res in structure.get_residues():
        predicted_coords.append(torch.Tensor(res["N"].get_coord()))
        predicted_coords.append(torch.Tensor(res["CA"].get_coord()))
        predicted_coords.append(torch.Tensor(res["C"].get_coord()))
    return torch.stack(predicted_coords).view(-1, 9)


def get_backbone_positions_from_angular_prediction(
    angular_emissions, batch_sizes, device
):
    # angular_emissions -1 x minibatch size x 3 (omega, phi, psi)
    points = pnerf.dihedral_to_point(angular_emissions, device)
    coordinates = (
        pnerf.point_to_coordinate(points, device) / 100
    )  # devide by 100 to angstrom unit
    return (
        coordinates.transpose(0, 1)
        .contiguous()
        .view(len(batch_sizes), -1, 9)
        .transpose(0, 1),
        batch_sizes,
    )


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


def encode_primary_string(primary):
    return list([AA_ID_DICT[aa] for aa in primary])


def intial_pos_from_aa_string(batch_aa_string):
    structures = []
    for aa_string in batch_aa_string:
        structure = get_structure_from_angles(
            aa_string,
            np.repeat([-120], len(aa_string) - 1),
            np.repeat([140], len(aa_string) - 1),
            np.repeat([-370], len(aa_string) - 1),
        )
        structures.append(structure)
    return structures


def pass_messages(aa_features, message_transformation, device):
    # aa_features (#aa, #features) - each row represents the amino acid type
    # (embedding) and the positions of the backbone atoms
    # message_transformation: (-1 * 2 * feature_size) ->
    #                           (-1 * output message size)
    feature_size = aa_features.size(1)
    aa_count = aa_features.size(0)
    eye = (
        torch.eye(aa_count, dtype=torch.uint8)
        .view(-1)
        .expand(2, feature_size, -1)
        .transpose(1, 2)
        .transpose(0, 1)
        .device(device)
    )
    eye_inverted = torch.ones(eye.size(), dtype=torch.uint8, device=device) - eye
    features_repeated = aa_features.repeat((aa_count, 1)).view(
        (aa_count, aa_count, feature_size)
    )
    aa_messages = (
        torch.stack((features_repeated.transpose(0, 1), features_repeated))
        .transpose(0, 1)
        .transpose(1, 2)
        .view(-1, 2, feature_size)
    )
    # (aa_count^2 - aa_count) x 2 x aa_features
    # (all pairs except for reflexive connections)
    aa_msg_pairs = torch.masked_select(aa_messages, eye_inverted).view(
        -1, 2, feature_size
    )
    transformed = message_transformation(aa_msg_pairs).view(aa_count, aa_count - 1, -1)
    transformed_sum = transformed.sum(dim=1)  # aa_count x output message size
    return transformed_sum


def masked_select(data, mask, X=None):
    """
    This masked_select works slightly differently.
    In the mask, there'll be a chain of 0s, all of these are not selected
    Instead, they are replaced by a single value defined by X
    This shows that the protein is discontinuous and we should not consider all
    the amino acids as continuous after masking
    Eg: data = [A, C, D, E, F, G, H, I, K, L, M, N] => Assume all are chars 'A'
        mask = [1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1]
        X = 'X'
        output = [A, C, X, G, H, I, X, L, M, N]
        Without X, the output would mean that C and G are adjacent and therefore the
        calculations of backbone angles will go wrong
    """
    output = []
    for i, val in enumerate(mask):
        if val == 1:
            if i != 0 and mask[i - 1] == 0 and X is not None:
                output.append(X)
            output.append(data[i])
    return np.array(output)


def calculate_dihedral_from_points(points):
    # Source: https://math.stackexchange.com/questions/47059/how-do-i-calculate-a-dihedral-angle-given-cartesian-coordinates
    b = np.zeros((3, 3))
    n = np.zeros((2, 3))
    for i in range(1, 4):
        b[i - 1] = points[i] - points[i - 1]
    for i in range(1, 3):
        tmp = np.cross(b[i - 1], b[i])
        try:
            n[i - 1] = tmp / np.linalg.norm(tmp)
        except RuntimeWarning:
            print("M", end="")
    m = np.cross(n[0], b[1] / np.linalg.norm(b[1]))
    x = np.dot(n[0], n[1])
    y = np.dot(m, n[1])
    return -np.arctan2(y, x)


def calculate_phi_from_masked_tertiary(tertiary_masked):
    flg = 0
    phi = []
    for i, aa in enumerate(tertiary_masked):
        # If there were missing residues
        if np.allclose(aa, np.zeros(9)):
            flg = 0
            continue
        if flg == 0:
            flg = 1
            phi.append(0)
            continue
        points = np.zeros((4, 3))
        points[0] = tertiary_masked[i - 1][6:]
        points[1:] = np.reshape(aa, (3, 3))
        phi.append(calculate_dihedral_from_points(points))
    return np.array(phi) * 180.0 / np.pi


def calculate_psi_from_masked_tertiary(tertiary_masked):
    flg = 1
    psi = []
    sz = len(tertiary_masked)
    for i, aa in enumerate(tertiary_masked):
        # If there were missing residues
        if i + 1 == sz or np.allclose(tertiary_masked[i + 1], np.zeros(9)):
            flg = 0
            psi.append(0)
            continue
        if flg == 0:
            flg = 1
            continue
        points = np.zeros((4, 3))
        points[0:3] = np.reshape(aa, (3, 3))
        points[3] = tertiary_masked[i + 1][:3]
        psi.append(calculate_dihedral_from_points(points))
    return np.array(psi) * 180.0 / np.pi
