import logging
import multiprocessing
from datetime import timedelta
from os import listdir, makedirs
from os.path import exists
from shutil import rmtree
from time import time

import numpy as np

from constants import (
    AA_ID_DICT,
    DSSP_DICT,
    FORCE_PREPROCESSING_OVERWRITE,
    MASK_DICT,
    PREPROCESS_PROTEIN_WITH_MISSING_RESIDUES,
)
from util import (
    calculate_omega_from_masked_tertiary,
    calculate_phi_from_masked_tertiary,
    calculate_psi_from_masked_tertiary,
    masked_select,
)


def read_protein(file_pointer):
    protein = {}

    is_protein_info_correct = True

    while True:
        next_line = file_pointer.readline()

        if not is_protein_info_correct:
            if next_line == "\n":
                return {}
            else:
                continue

        # ID of the protein
        if next_line == "[ID]\n":
            id_ = file_pointer.readline()[:-1]
            protein.update({"id": id_})

        # Amino acid sequence of the protein
        elif next_line == "[PRIMARY]\n":
            # Convert amino acids into their numeric representation
            primary = list([AA_ID_DICT[aa] for aa in file_pointer.readline()[:-1]])
            seq_len = len(primary)
            protein.update({"primary": primary})

        # PSSM matrix + Information Content
        # Dimensions: [21, Protein Length]
        # First 20 rows represents the PSSM info of
        # each amino acid in alphabetical order
        # 21st row represents information content
        elif next_line == "[EVOLUTIONARY]\n":
            evolutionary = []
            for residue in range(21):
                evolutionary.append(
                    [
                        np.float32(log_likelihoods)
                        for log_likelihoods in file_pointer.readline().split()
                    ]
                )
                if len(evolutionary[-1]) != seq_len:
                    logging.info(
                        "Error in evolutionary information of protein with id %s. Skipping it",
                        id_,
                    )
                    is_protein_info_correct = False
                    continue
            protein.update({"evolutionary": evolutionary})

        # Secondary structure information of the protein
        # 8 classes: L, H, B, E, G, I, T, S
        elif next_line == "[SECONDARY]\n":
            secondary = list([DSSP_DICT[dssp] for dssp in file_pointer.readline()[:-1]])
            protein.update({"secondary": secondary})

        # Tertiary structure information of the protein
        # The values are represented in picometers
        # => Relative to PDB, multiply by 100
        # Dimensions: [3, 3 * Protein Length]
        # Eg: for protein of length 1
        #      N       C_a       C
        # X  2841.8,  2873.4,  2919.7
        # Y  -864.7,  -957.9,  -877.0
        # Z -6727.1, -6616.3, -6496.2
        elif next_line == "[TERTIARY]\n":
            tertiary = []
            for axis in range(3):
                tertiary.append(
                    [
                        np.float32(coord) / 100.0
                        for coord in file_pointer.readline().split()
                    ]
                )
                if len(tertiary[-1]) != 3 * seq_len:
                    logging.info(
                        "Error in tertiary information of protein with id %s. Skipping it",
                        id_,
                    )
                    is_protein_info_correct = False
                    continue
            protein.update({"tertiary": tertiary})

        # Some residues might be missing from a protein
        # Mask contains a '+' or a '-' based on whether
        # the residue has tertiary information or not
        elif next_line == "[MASK]\n":
            mask = list([MASK_DICT[aa] for aa in file_pointer.readline()[:-1]])
            if len(mask) != seq_len:
                logging.info(
                    "Error in masking information of protein with id %s. Skipping it",
                    id_,
                )
                is_protein_info_correct = False
                continue
            protein.update({"mask": mask})

        # All the information of the current protein
        # is available in dict now
        elif next_line == "\n":
            return protein

        # File has been read completely
        elif next_line == "":
            return None


def process_protein(protein):

    if protein == {}:
        return None

    # Preprocess all the proteins that do not have missing residues
    if not PREPROCESS_PROTEIN_WITH_MISSING_RESIDUES:
        skip_flg = 0
        prv = 0
        for ind, el in enumerate(np.argwhere(protein["mask"])):
            if ind != 0 and el != prv + 1:
                skip_flg = 1
                break
            prv = el

        if skip_flg:
            return None

    primary_masked = np.array(
        masked_select(protein["primary"], protein["mask"]), dtype=np.uint8
    )

    evolutionary_transposed = np.array(protein["evolutionary"]).T
    evolutionary_masked = np.array(
        masked_select(evolutionary_transposed, protein["mask"]), dtype=np.float32
    )

    # secondary_masked = np.array(
    #     masked_select(protein["secondary"], protein["mask"]), dtype=np.uint8
    # )

    # protein["tertiary"] is of shape [3, 3 * Length]
    tertiary_transposed = np.array(protein["tertiary"]).T
    tertiary_reshaped = np.reshape(tertiary_transposed, (-1, 9))

    # X is np.zeros(9) for tertiary info
    # Putting X as a signal of missing residues
    # tertiary_masked is of shape [Masked_Length + X_count, 9]
    tertiary_masked = np.array(
        masked_select(tertiary_reshaped, protein["mask"], np.zeros(9)), dtype=np.float32
    )

    # There are problems with some of the proteins in the dataset
    # Skip if that is the case

    # TODO Make it such that only the residue is missing and don't
    # skip the whole protein
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
        return None

    # logging.info(protein["id"])
    masked_length = len(primary_masked)
    assert len(evolutionary_masked) == masked_length
    # assert len(secondary_masked) == masked_length
    phi = calculate_phi_from_masked_tertiary(tertiary_masked)
    assert len(phi) == masked_length
    psi = calculate_psi_from_masked_tertiary(tertiary_masked)
    assert len(psi) == masked_length
    omega = calculate_omega_from_masked_tertiary(tertiary_masked)
    assert len(omega) == masked_length

    processed_protein = {
        "id": protein["id"],
        "primary": primary_masked,
        "evolutionary": evolutionary_masked,
        "phi": phi,
        "psi": psi,
        "omega": omega,
        "tertiary": tertiary_masked,
    }
    return processed_protein


def process_file(input_file, output_folder):
    logging.info("Processing raw data file %s", input_file)
    input_file_pointer = open("data/raw/" + input_file, "r")
    idx = 0
    total_read_time = 0
    total_process_time = 0
    total_write_time = 0
    prev_read_time = 0
    prev_process_time = 0
    prev_write_time = 0
    num_cpus = multiprocessing.cpu_count()

    with multiprocessing.Pool(num_cpus) as p:
        while True:
            proteins = []
            # While there's more proteins to process
            read_time_start = time()
            for _ in range(num_cpus):
                protein = read_protein(input_file_pointer)
                # Reached end of file, stop processing
                proteins.append(protein)
                if protein is None:
                    break
            idx += num_cpus
            total_read_time += time() - read_time_start

            # Process the protein
            process_time_start = time()
            processed_proteins = p.map(process_protein, proteins)
            total_process_time += time() - process_time_start

            # The .npz file format is a zipped archive of files
            # named after the variables they contain. The archive is
            # NOT compressed and each file in the archive contains
            # one variable in .npy format
            write_time_start = time()
            for protein in processed_proteins:
                # If there was an error in the protein, skip it
                if protein is None:
                    continue
                np.savez(output_folder + protein["id"] + ".npz", **protein)
            total_write_time += time() - write_time_start

            if proteins[-1] is None:
                break

            if idx % 1000 == 0:
                logging.info("Last: %s | %d proteins processed", proteins[-1]["id"], idx)
                logging.info(
                    "Read time %s | Process time %s | Write time %s",
                    str(timedelta(seconds=total_read_time - prev_read_time)),
                    str(timedelta(seconds=total_process_time - prev_process_time)),
                    str(timedelta(seconds=total_write_time - prev_write_time)),
                )
                prev_read_time = total_read_time
                prev_process_time = total_process_time
                prev_write_time = total_write_time

        input_file_pointer.close()
        logging.info("Wrote output of %d proteins to %s folder", idx, output_folder)
        logging.info(
            "Total Read time %s | Process time %s | Write time %s",
            str(timedelta(seconds=total_read_time)),
            str(timedelta(seconds=total_process_time)),
            str(timedelta(seconds=total_write_time)),
        )


def filter_input_files(input_files):
    """
    Returns a list of files that do not have the provided
    file_endings from the provided list of files
    """
    disallowed_file_endings = (".gitignore", ".DS_Store", ".git")
    return list(filter(lambda x: not x.endswith(disallowed_file_endings), input_files))


def preprocess_raw_data():
    logging.info("Starting pre-processing of raw data...")

    input_files = listdir("data/raw/")
    input_files_filtered = filter_input_files(input_files)
    for filename in input_files_filtered:
        preprocessed_folder_path = "data/preprocessed/" + filename + "/"
        if not PREPROCESS_PROTEIN_WITH_MISSING_RESIDUES:
            preprocessed_folder_path = preprocessed_folder_path[:-1] + "_no_missing/"
        if FORCE_PREPROCESSING_OVERWRITE:
            rmtree(preprocessed_folder_path)
        if not exists(preprocessed_folder_path):
            makedirs(preprocessed_folder_path)
            # Careful while saving tertiary info for proteins with missing residues
            process_file(filename, preprocessed_folder_path)
        else:
            logging.info(
                """
                Preprocessed files already present in %s directory.
                Use --force-pre-processing-overwrite or delete the
                folder manually to overwrite
                """,
                preprocessed_folder_path,
            )

    logging.info("Completed pre-processing.")


if __name__ == "__main__":
    from sys import stdout

    logging.basicConfig(
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
        stream=stdout,
    )
    preprocess_raw_data()
