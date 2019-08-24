import numpy as np


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
