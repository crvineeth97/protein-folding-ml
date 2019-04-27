# This file is part of the OpenProtein project.
#
# @author Jeppe Hallgren
#
# For license information, please see the LICENSE file in the root directory.

import glob
import os
import os.path

import h5py
import numpy as np
import torch

from util import (
    calculate_dihedral_angles_over_minibatch,
    encode_primary_string,
    get_backbone_positions_from_angular_prediction,
)

MAX_SEQUENCE_LENGTH = 2000


def read_protein_from_file(file_pointer):

    dict_ = {}
    _dssp_dict = {"L": 0, "H": 1, "B": 2, "E": 3, "G": 4, "I": 5, "T": 6, "S": 7}
    _mask_dict = {"-": 0, "+": 1}
    is_protein_info_correct = True

    while True:
        if not is_protein_info_correct:
            return {}
        next_line = file_pointer.readline()
        if next_line == "[ID]\n":
            # ID of the protein
            id_ = file_pointer.readline()[:-1]
            dict_.update({"id": id_})
        elif next_line == "[PRIMARY]\n":
            # Amino acid sequence of the protein
            primary = encode_primary_string(file_pointer.readline()[:-1])
            seq_len = len(primary)
            dict_.update({"primary": primary})
            # dict_.update({'sequence_length': seq_len})
        elif next_line == "[EVOLUTIONARY]\n":
            # PSSM matrix + Information Content
            # Dimensions: [21, Protein Length]
            # First 20 rows represents the PSSM info of
            # each amino acid in alphabetical order
            # 21st row represents information content
            evolutionary = []
            for residue in range(21):
                evolutionary.append(
                    [float(step) for step in file_pointer.readline().split()]
                )
                if len(evolutionary[-1]) != seq_len:
                    print(
                        "Error in evolutionary information of protein with id "
                        + id_
                        + ". Skipping it"
                    )
                    is_protein_info_correct = False
                    break
            dict_.update({"evolutionary": evolutionary})
        elif next_line == "[SECONDARY]\n":
            secondary = list(
                [_dssp_dict[dssp] for dssp in file_pointer.readline()[:-1]]
            )
            dict_.update({"secondary": secondary})
        elif next_line == "[TERTIARY]\n":
            tertiary = []
            # The values are represented in picometers
            # => Relative to PDB, multiply by 100
            # Dimensions: [3, 3 * Protein Length]
            # Eg: for protein of length 1
            #      N       C_a       C
            # X  2841.8,  2873.4,  2919.7
            # Y  -864.7,  -957.9,  -877.0
            # Z -6727.1, -6616.3, -6496.2
            for axis in range(3):
                tertiary.append(
                    [float(coord) for coord in file_pointer.readline().split()]
                )
                if len(tertiary[-1]) != 3 * seq_len:
                    print(
                        "Error in tertiary information of protein with id "
                        + id_
                        + ". Skipping it"
                    )
                    is_protein_info_correct = False
                    break
            dict_.update({"tertiary": tertiary})
        elif next_line == "[MASK]\n":
            mask = list([_mask_dict[aa] for aa in file_pointer.readline()[:-1]])
            if len(mask) != seq_len:
                print(
                    "Error in masking information of protein with id "
                    + id_
                    + ". Skipping it"
                )
                is_protein_info_correct = False
            dict_.update({"mask": mask})
        elif next_line == "\n":
            return dict_
        elif next_line == "":
            return None


def process_file(input_file, output_file, use_gpu):
    print("Processing raw data file", input_file)
    # Means resize every 50 proteins
    mini_batch_size = 50
    idx = 0
    # create output file
    f = h5py.File(output_file, "w")
    dset1 = f.create_dataset(
        "primary",
        (mini_batch_size, MAX_SEQUENCE_LENGTH),
        maxshape=(None, MAX_SEQUENCE_LENGTH),
        dtype="int32",
    )
    dset2 = f.create_dataset(
        "tertiary",
        (mini_batch_size, MAX_SEQUENCE_LENGTH, 9),
        maxshape=(None, MAX_SEQUENCE_LENGTH, 9),
        dtype="float",
    )
    dset3 = f.create_dataset(
        "mask",
        (mini_batch_size, MAX_SEQUENCE_LENGTH),
        maxshape=(None, MAX_SEQUENCE_LENGTH),
        dtype="uint8",
    )

    input_file_pointer = open("data/raw/" + input_file, "r")

    while True:
        # while there's more proteins to process
        next_protein = read_protein_from_file(input_file_pointer)
        if next_protein == {}:
            continue
        if next_protein is None:
            break

        sequence_length = len(next_protein["primary"])

        if sequence_length > MAX_SEQUENCE_LENGTH:
            print("Dropping protein as length is too long:", sequence_length)
            continue

        primary_padded = np.zeros(MAX_SEQUENCE_LENGTH)
        tertiary_padded = np.zeros((9, MAX_SEQUENCE_LENGTH))
        mask_padded = np.zeros(MAX_SEQUENCE_LENGTH)

        primary_padded[:sequence_length] = next_protein["primary"]
        t_transposed = np.ravel(np.array(next_protein["tertiary"]).T)
        t_reshaped = np.reshape(t_transposed, (sequence_length, 9)).T

        tertiary_padded[:, :sequence_length] = t_reshaped
        mask_padded[:sequence_length] = next_protein["mask"]

        # The masked_select concats 2 unrelated amino acids
        # and their dihedral angles are calculated, which should
        # not be the case.
        # TODO Figure out a way to solve this
        # mask = torch.Tensor(mask_padded).type(dtype=torch.uint8)
        # prim = torch.masked_select(torch.Tensor(
        #     primary_padded).type(dtype=torch.long), mask)
        # Broadcasting works so masking will apply for all 9 rows
        # Unsqueeze adds a 1 dimension at specified position
        # Done so that the PNERF can work properly
        # pos = torch.masked_select(torch.Tensor(tertiary_padded), mask).view(
        #     9, -1).transpose(0, 1).unsqueeze(1) / 100
        # Dimensions of pos: [Protein Length, 1, 9]

        dset1[idx] = primary_padded
        dset2[idx] = tertiary_padded
        dset3[idx] = mask_padded
        idx += 1

        if idx % mini_batch_size == 0:
            resize_shape = (idx / mini_batch_size + 1) * mini_batch_size
            dset1.resize((resize_shape, MAX_SEQUENCE_LENGTH))
            dset2.resize((resize_shape, MAX_SEQUENCE_LENGTH, 9))
            dset3.resize((resize_shape, MAX_SEQUENCE_LENGTH))

    print("Wrote output of ", idx + 1, " proteins to ", output_file)


def filter_input_files(input_files):
    disallowed_file_endings = (".gitignore", ".DS_Store")
    return list(filter(lambda x: not x.endswith(disallowed_file_endings), input_files))


def process_raw_data(use_gpu, force_pre_processing_overwrite=False):
    print("Starting pre-processing of raw data...")
    input_files = glob.glob("data/raw/*")
    input_files_filtered = filter_input_files(input_files)
    for file_path in input_files_filtered:
        filename = file_path.split("/")[-1]
        preprocessed_file_name = "data/preprocessed/" + filename + ".hdf5"

        # check if we should remove the any previously processed files
        if os.path.isfile(preprocessed_file_name):
            print("Preprocessed file for " + filename + " already exists.")
            if force_pre_processing_overwrite:
                print(
                    "force_pre_processing_overwrite flag set to True, overwriting old file..."
                )
                os.remove(preprocessed_file_name)
            else:
                print("Skipping pre-processing for this file...")

        if not os.path.isfile(preprocessed_file_name):
            process_file(filename, preprocessed_file_name, use_gpu)
    print("Completed pre-processing.")
