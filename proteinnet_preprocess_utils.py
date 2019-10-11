import numpy as np
import logging
from constants import AA_ID_DICT, DSSP_DICT, MASK_DICT


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
            print("Error", end="")
    m = np.cross(n[0], b[1] / np.linalg.norm(b[1]))
    x = np.dot(n[0], n[1])
    y = np.dot(m, n[1])
    return -np.arctan2(y, x)


def calculate_phi(tertiary):
    """
    The phi angle of an amino acid at the i'th position is calculated
    using the coordinates of the C{i-1}, N{i}, Ca{i}, C{i} atoms. Hence,
    the phi angle of the first amino acid is always 0
    """
    is_previous_aa_present = False
    phi = []
    for i, aa in enumerate(tertiary):
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
        points[0] = tertiary[i - 1][6:]
        points[1:] = np.reshape(aa, (3, 3))
        phi.append(calculate_dihedral_from_points(points))
    return np.array(phi, dtype=np.float32)


def calculate_psi(tertiary):
    """
    The psi angle of an amino acid at the i'th position is calculated
    using the coordinates of the N{i}, Ca{i}, C{i}, N{i+1} atoms. Hence,
    the psi angle of the last amino acid is always 0
    """
    is_next_aa_present = True
    psi = []
    protein_length = len(tertiary)
    for i, aa in enumerate(tertiary):
        # If there were missing residues, we don't have their tertiary
        # positions and hence psi will be 0 for the last amino acid
        # before the missing residues, because we don't have N{i+1}
        if i + 1 == protein_length or np.allclose(tertiary[i + 1], np.zeros(9)):
            is_next_aa_present = False
            psi.append(0)
            continue
        if not is_next_aa_present:
            is_next_aa_present = True
            continue
        points = np.zeros((4, 3))
        points[:3] = np.reshape(aa, (3, 3))
        points[3] = tertiary[i + 1][:3]
        psi.append(calculate_dihedral_from_points(points))
    return np.array(psi, dtype=np.float32)


def calculate_omega(tertiary):
    """
    The omega angle of an amino acid at the i'th position is calculated
    using the coordinates of the Ca{i}, C{i}, N{i+1}, Ca{i+1} atoms. Hence,
    the omega angle of the last amino acid is always 0
    """
    is_next_aa_present = True
    omega = []
    protein_length = len(tertiary)
    for i, aa in enumerate(tertiary):
        # If there were missing residues, we don't have their tertiary
        # positions and hence omega will be 0 for the last amino acid
        # before the missing residues, because we don't have C{i-1}
        if i + 1 == protein_length or np.allclose(tertiary[i + 1], np.zeros(9)):
            is_next_aa_present = False
            omega.append(0)
            continue
        if not is_next_aa_present:
            is_next_aa_present = True
            continue
        points = np.zeros((4, 3))
        points[0:2] = np.reshape(aa[3:], (2, 3))
        points[2:] = np.reshape(tertiary[i + 1][:6], (2, 3))
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


def build_contact_map(tertiary):
    protein_length = len(tertiary)
    contact_map = np.zeros((protein_length, protein_length), dtype=np.float32)
    for i in range(protein_length):
        for j in range(i + 1, protein_length):
            dist = np.linalg.norm(tertiary[i][3:6] - tertiary[j][3:6])
            contact_map[i][j] = contact_map[j][i] = dist
    return contact_map


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
