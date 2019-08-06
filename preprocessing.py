import logging
from os import listdir, makedirs
from os.path import exists
from shutil import rmtree

import numpy as np

from constants import (
    AA_ID_DICT,
    DSSP_DICT,
    FORCE_PREPROCESSING_OVERWRITE,
    MASK_DICT,
    PREPROCESS_PROTEIN_WITH_MISSING_RESIDUES,
)


def encode_primary_string(primary):
    return list([AA_ID_DICT[aa] for aa in primary])


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
            logging.info("Error", end="")
    m = np.cross(n[0], b[1] / np.linalg.norm(b[1]))
    x = np.dot(n[0], n[1])
    y = np.dot(m, n[1])
    return -np.arctan2(y, x)


def calculate_phi_from_masked_tertiary(tertiary_masked):
    """
    The phi angle of an amino acid at the i'th position is calculated
    using the coordinates of the C{i-1}, N{i}, Ca{i}, C{i} atoms. Hence,
    the phi angle of the first amino acid is always 0
    """
    is_previous_aa_present = False
    phi = []
    for i, aa in enumerate(tertiary_masked):
        # If there were missing residues, we don't have their tertiary
        # positions and hence phi will be 0 for the first amino acid
        # after the missing residues, because we don't have C{i-1}
        if np.allclose(aa, np.zeros(9)):
            is_previous_aa_present = False
            continue
        if not is_previous_aa_present:
            is_previous_aa_present = True
            phi.append(0)
            continue
        points = np.zeros((4, 3))
        points[0] = tertiary_masked[i - 1][6:]
        points[1:] = np.reshape(aa, (3, 3))
        phi.append(calculate_dihedral_from_points(points))
    return np.array(phi, dtype=np.float32)


def calculate_psi_from_masked_tertiary(tertiary_masked):
    """
    The psi angle of an amino acid at the i'th position is calculated
    using the coordinates of the N{i}, Ca{i}, C{i}, N{i+1} atoms. Hence,
    the psi angle of the last amino acid is always 0
    """
    is_next_aa_present = True
    psi = []
    sz = len(tertiary_masked)
    for i, aa in enumerate(tertiary_masked):
        # If there were missing residues, we don't have their tertiary
        # positions and hence psi will be 0 for the last amino acid
        # before the missing residues, because we don't have N{i+1}
        if i + 1 == sz or np.allclose(tertiary_masked[i + 1], np.zeros(9)):
            is_next_aa_present = False
            psi.append(0)
            continue
        if not is_next_aa_present:
            is_next_aa_present = True
            continue
        points = np.zeros((4, 3))
        points[:3] = np.reshape(aa, (3, 3))
        points[3] = tertiary_masked[i + 1][:3]
        psi.append(calculate_dihedral_from_points(points))
    return np.array(psi, dtype=np.float32)


def calculate_omega_from_masked_tertiary(tertiary_masked):
    """
    The omega angle of an amino acid at the i'th position is calculated
    using the coordinates of the Ca{i}, C{i}, N{i+1}, Ca{i+1} atoms. Hence,
    the omega angle of the last amino acid is always 0
    """
    is_next_aa_present = True
    omega = []
    sz = len(tertiary_masked)
    for i, aa in enumerate(tertiary_masked):
        # If there were missing residues, we don't have their tertiary
        # positions and hence omega will be 0 for the last amino acid
        # before the missing residues, because we don't have C{i-1}
        if i + 1 == sz or np.allclose(tertiary_masked[i + 1], np.zeros(9)):
            is_next_aa_present = False
            omega.append(0)
            continue
        if not is_next_aa_present:
            is_next_aa_present = True
            continue
        points = np.zeros((4, 3))
        points[0:2] = np.reshape(aa[3:], (2, 3))
        points[2:] = np.reshape(tertiary_masked[i + 1][:6], (2, 3))
        omega.append(calculate_dihedral_from_points(points))
    return np.array(omega, dtype=np.float32)


def masked_select(data, mask, X=None):
    """
    This masked_select works slightly differently.
    In the mask, there'll be a chain of 0s, all of these are not selected
    Instead, they are replaced by a single value defined by X
    This shows that the protein is discontinuous and we should not consider all
    the amino acids as continuous after masking
    Eg: data = [A, C, D, E, F, G, H, I, K, L, M, N] => Assume all are chars 'A'
        mask = [0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0]
        X = 'X'
        output = [D, X, G, H, I, X, L, M]
        Without X, the output would mean that C and G are adjacent and therefore the
        calculations of backbone angles will go wrong
    The above is just an example and this function is not applied to the primary sequence
    """
    output = []
    for i, val in enumerate(mask):
        if val == 1:
            if i != 0 and mask[i - 1] == 0 and X is not None:
                output.append(X)
            output.append(data[i])

    #  If there is an X at the beginning, remove it
    if X is not None and np.allclose(output[0], X):
        output = output[1:]

    return output


def read_protein(file_pointer):
    dict_ = {}

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
            dict_.update({"id": id_})

        # Amino acid sequence of the protein
        elif next_line == "[PRIMARY]\n":
            # Convert amino acids into their numeric representation
            primary = encode_primary_string(file_pointer.readline()[:-1])
            seq_len = len(primary)
            dict_.update({"primary": primary})

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
            dict_.update({"evolutionary": evolutionary})

        # Secondary structure information of the protein
        # 8 classes: L, H, B, E, G, I, T, S
        elif next_line == "[SECONDARY]\n":
            secondary = list([DSSP_DICT[dssp] for dssp in file_pointer.readline()[:-1]])
            dict_.update({"secondary": secondary})

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
            dict_.update({"tertiary": tertiary})

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
            dict_.update({"mask": mask})

        # All the information of the current protein
        # is available in dict now
        elif next_line == "\n":
            return dict_

        # File has been read completely
        elif next_line == "":
            return None


def process_file(input_file, output_folder, save_tertiary):
    logging.info("Processing raw data file %s", input_file)
    input_file_pointer = open("data/raw/" + input_file, "r")
    idx = 0

    while True:
        # While there's more proteins to process
        protein = read_protein(input_file_pointer)

        # If there was an error in the protein, skip it
        if protein == {}:
            continue

        # Reached end of file, stop processing
        if protein is None:
            break

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
                continue

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
            masked_select(tertiary_reshaped, protein["mask"], np.zeros(9)),
            dtype=np.float32,
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
            continue

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

        if save_tertiary:
            assert len(tertiary_masked) == masked_length
            np.savez(
                output_folder + protein["id"] + ".npz",
                primary=primary_masked,
                evolutionary=evolutionary_masked,
                phi=phi,
                psi=psi,
                omega=omega,
                tertiary=tertiary_masked,
            )
        else:
            np.savez(
                output_folder + protein["id"] + ".npz",
                primary=primary_masked,
                evolutionary=evolutionary_masked,
                phi=phi,
                psi=psi,
                omega=omega,
            )
        idx += 1
        if idx % 1000 == 0:
            logging.info("Last: %s - %d proteins processed", protein["id"], idx)

    input_file_pointer.close()
    logging.info("Wrote output of %d proteins to %s folder", idx, output_folder)


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
            if "validation" in filename or "testing" in filename:
                # Careful while saving tertiary info for proteins with missing residues
                process_file(filename, preprocessed_folder_path, True)
            else:
                process_file(filename, preprocessed_folder_path, False)
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
