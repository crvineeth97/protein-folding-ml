import logging

# import multiprocessing
from datetime import timedelta
from os import listdir, makedirs
from os.path import exists
from shutil import rmtree
from time import time

import numpy as np

from constants import (
    FORCE_PREPROCESSING_OVERWRITE,
    PREPROCESS_PROTEIN_WITH_MISSING_RESIDUES,
)
from preprocess_utils import (
    calculate_omega,
    calculate_phi,
    calculate_psi,
    calculate_binary_contact_map,
    masked_select,
    read_protein,
)


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
    phi = calculate_phi(tertiary_masked)
    assert len(phi) == masked_length
    psi = calculate_psi(tertiary_masked)
    assert len(psi) == masked_length
    omega = calculate_omega(tertiary_masked)
    assert len(omega) == masked_length
    contact_map = calculate_binary_contact_map(tertiary_masked)
    assert len(contact_map) == masked_length
    assert len(contact_map[0]) == masked_length

    processed_protein = {
        "id": protein["id"],
        "primary": primary_masked,
        "evolutionary": evolutionary_masked,
        "phi": phi,
        "psi": psi,
        "omega": omega,
        "tertiary": tertiary_masked,
        "contact_map": contact_map,
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
    # num_cpus = multiprocessing.cpu_count()
    # logging.info("Using %d CPUs for preprocessing", num_cpus)

    # with multiprocessing.Pool(num_cpus) as pool:
    while True:
        # proteins = []
        # While there's more proteins to process
        read_time_start = time()
        # for _ in range(num_cpus):
        protein = read_protein(input_file_pointer)
        # Reached end of file, stop processing
        # proteins.append(protein)
        if protein is None:
            break
        idx += 1
        total_read_time += time() - read_time_start

        # Process the protein
        process_time_start = time()
        protein = process_protein(protein)
        total_process_time += time() - process_time_start

        # The .npz file format is a zipped archive of files
        # named after the variables they contain. The archive is
        # NOT compressed and each file in the archive contains
        # one variable in .npy format
        write_time_start = time()
        # for protein in processed_proteins:
        # If there was an error in the protein, skip it
        if protein is None:
            continue
        np.savez(output_folder + protein["id"] + ".npz", **protein)
        total_write_time += time() - write_time_start

        if idx % 1000 == 0:
            logging.info("Last: %s | %d proteins processed", protein["id"], idx)
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

    file_handler = logging.FileHandler(filename="output/" + "preprocessing.txt")
    stdout_handler = logging.StreamHandler(stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
        handlers=handlers,
    )
    preprocess_raw_data()
