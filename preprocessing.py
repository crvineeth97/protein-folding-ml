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

from util import (
    DSSP_DICT,
    MASK_DICT,
    calculate_phi_from_masked_tertiary,
    calculate_psi_from_masked_tertiary,
    encode_primary_string,
    masked_select,
)

MAX_SEQUENCE_LENGTH = 2000


def read_protein_from_file(file_pointer):

    dict_ = {}

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
            secondary = list([DSSP_DICT[dssp] for dssp in file_pointer.readline()[:-1]])
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
            mask = list([MASK_DICT[aa] for aa in file_pointer.readline()[:-1]])
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


def process_file(input_file, output_file):
    print("Processing raw data file", input_file)
    # Means resize every 50 proteins
    mini_batch_size = 50
    idx = 0
    # create output file
    f = h5py.File(output_file, "w")
    # The following variables are datasets of the h5py file
    # Can add more information if available
    dsetl = f.create_dataset(
        "length", (mini_batch_size, 1), maxshape=(None, 1), dtype="int32"
    )
    dset1 = f.create_dataset(
        "primary",
        (mini_batch_size, MAX_SEQUENCE_LENGTH),
        maxshape=(None, MAX_SEQUENCE_LENGTH),
        dtype="uint8",
    )
    dset2 = f.create_dataset(
        "evolutionary",
        (mini_batch_size, MAX_SEQUENCE_LENGTH, 21),
        maxshape=(None, MAX_SEQUENCE_LENGTH, 21),
        dtype="float",
    )
    # No secondary information available in ProteinNet
    # Skipping this for now
    # dset3 = f.create_dataset(
    #     "secondary",
    #     (mini_batch_size, MAX_SEQUENCE_LENGTH),
    #     maxshape=(None, MAX_SEQUENCE_LENGTH),
    #     dtype="uint8",
    # )
    # Instead of storing tertiary data, let us store the phi, psi angles
    # dset4 = f.create_dataset(
    #     "tertiary",
    #     (mini_batch_size, MAX_SEQUENCE_LENGTH, 9),
    #     maxshape=(None, MAX_SEQUENCE_LENGTH, 9),
    #     dtype="float",
    # )
    dset4 = f.create_dataset(
        "phi",
        (mini_batch_size, MAX_SEQUENCE_LENGTH),
        maxshape=(None, MAX_SEQUENCE_LENGTH),
        dtype="float",
    )
    dset5 = f.create_dataset(
        "psi",
        (mini_batch_size, MAX_SEQUENCE_LENGTH),
        maxshape=(None, MAX_SEQUENCE_LENGTH),
        dtype="float",
    )
    # No need to store the masks
    # dset5 = f.create_dataset(
    #     "mask",
    #     (mini_batch_size, MAX_SEQUENCE_LENGTH),
    #     maxshape=(None, MAX_SEQUENCE_LENGTH),
    #     dtype="uint8",
    # )

    input_file_pointer = open("data/raw/" + input_file, "r")

    while True:
        # While there's more proteins to process
        protein = read_protein_from_file(input_file_pointer)
        if protein == {}:
            continue
        if protein is None:
            break

        sequence_length = len(protein["primary"])

        if sequence_length > MAX_SEQUENCE_LENGTH:
            print(
                "Protein "
                + protein["id"]
                + " is too large and will not be considered. Length = "
                + str(sequence_length)
            )
            continue
        primary_masked = masked_select(protein["primary"], protein["mask"])
        evolutionary_reshaped = np.array(protein["evolutionary"]).T
        evolutionary_masked = masked_select(evolutionary_reshaped, protein["mask"])
        # secondary_masked = masked_select(protein["secondary"], protein["mask"])
        tertiary_transposed = np.ravel(np.array(protein["tertiary"]).T)
        tertiary_reshaped = (
            np.reshape(tertiary_transposed, (sequence_length, 9)) / 100.0
        )
        # Putting np.zeros(9) as a signal of missing residues
        tertiary_masked = masked_select(tertiary_reshaped, protein["mask"], np.zeros(9))
        # If the first few residues were missing, no need to consider them
        if np.allclose(tertiary_masked[0], np.zeros(9)):
            tertiary_masked = tertiary_masked[1:]
        # There are problems with some of the proteins in the dataset
        # Skip if that is the case
        skip_flg = 0
        for coords in tertiary_masked:
            if not np.allclose(coords, np.zeros(9)) and (
                np.allclose(coords[:3], np.zeros(3))
                or np.allclose(coords[3:6], np.zeros(3))
                or np.allclose(coords[6:], np.zeros(3))
            ):
                skip_flg = 1
                break
        if skip_flg:
            continue

        # print(protein["id"])
        masked_length = len(primary_masked)

        primary_padded = np.zeros(MAX_SEQUENCE_LENGTH)
        evolutionary_padded = np.zeros((MAX_SEQUENCE_LENGTH, 21))
        # secondary_padded = np.zeros(MAX_SEQUENCE_LENGTH)
        phi_padded = np.zeros(MAX_SEQUENCE_LENGTH)
        psi_padded = np.zeros(MAX_SEQUENCE_LENGTH)

        primary_padded[:masked_length] = primary_masked
        evolutionary_padded[:masked_length] = evolutionary_masked
        assert len(evolutionary_masked) == masked_length
        # secondary_padded[:masked_length] = secondary_masked
        # assert len(secondary_masked) == masked_length
        phi = calculate_phi_from_masked_tertiary(tertiary_masked)
        assert len(phi) == masked_length
        phi_padded[:masked_length] = phi
        psi = calculate_psi_from_masked_tertiary(tertiary_masked)
        assert len(psi) == masked_length
        psi_padded[:masked_length] = psi

        dsetl[idx] = [masked_length]
        dset1[idx] = primary_padded
        dset2[idx] = evolutionary_padded
        # dset3[idx] = secondary_padded
        dset4[idx] = phi_padded
        dset5[idx] = psi_padded
        idx += 1

        if idx % mini_batch_size == 0:
            resize_shape = (idx / mini_batch_size + 1) * mini_batch_size
            dsetl.resize((resize_shape, 1))
            dset1.resize((resize_shape, MAX_SEQUENCE_LENGTH))
            dset2.resize((resize_shape, MAX_SEQUENCE_LENGTH, 21))
            # dset3.resize((resize_shape, MAX_SEQUENCE_LENGTH))
            dset4.resize((resize_shape, MAX_SEQUENCE_LENGTH))
            dset5.resize((resize_shape, MAX_SEQUENCE_LENGTH))

    input_file_pointer.close()
    print("Wrote output of ", idx, " proteins to ", output_file)


def filter_input_files(input_files):
    """
    Returns a list of files that do not have the provided
    file_endings from the provided list of files
    """
    disallowed_file_endings = (".gitignore", ".DS_Store")
    return list(filter(lambda x: not x.endswith(disallowed_file_endings), input_files))


def process_raw_data(force_pre_processing_overwrite=False):
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
            process_file(filename, preprocessed_file_name)

    print("Completed pre-processing.")
