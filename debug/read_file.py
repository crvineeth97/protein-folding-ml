import datetime
from os import listdir
from time import time

import numpy as np

# List of amino acids and their integer representation
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

# Secondary structure labels and their integer representation
DSSP_DICT = {"L": 0, "H": 1, "B": 2, "E": 3, "G": 4, "I": 5, "T": 6, "S": 7}

# Masking values and their integer representation
MASK_DICT = {"-": 0, "+": 1}


def encode_primary_string(primary):
    return list([AA_ID_DICT[aa] for aa in primary])


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
                    is_protein_info_correct = False
                    continue
            dict_.update({"tertiary": tertiary})

        # Some residues might be missing from a protein
        # Mask contains a '+' or a '-' based on whether
        # the residue has tertiary information or not
        elif next_line == "[MASK]\n":
            mask = list([MASK_DICT[aa] for aa in file_pointer.readline()[:-1]])
            if len(mask) != seq_len:
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


def process_file(input_file):
    input_file_pointer = open("data/raw/" + input_file, "r")
    while True:
        # While there's more proteins to process
        protein = read_protein(input_file_pointer)

        # If there was an error in the protein, skip it
        if protein == {}:
            continue

        # Reached end of file, stop processing
        if protein is None:
            break

    input_file_pointer.close()


def filter_input_files(input_files):
    """
    Returns a list of files that do not have the provided
    file_endings from the provided list of files
    """
    disallowed_file_endings = (".gitignore", ".DS_Store", ".git")
    return list(filter(lambda x: not x.endswith(disallowed_file_endings), input_files))


def preprocess_raw_data():
    input_files = listdir("data/raw/")
    input_files_filtered = filter_input_files(input_files)
    for filename in input_files_filtered:
        start = time()
        process_file(filename)
        print(
            "Time taken to read",
            filename,
            str(datetime.timedelta(seconds=time() - start)),
        )


if __name__ == "__main__":
    preprocess_raw_data()
