import numpy as np
from Bio.PDB import PDBList, PDBParser, is_aa


def calculate_dihedral_from_points(points):
    # Source: https://math.stackexchange.com/questions/47059/how-do-i-calculate-a-dihedral-angle-given-cartesian-coordinates
    b = np.zeros((3, 3))
    n = np.zeros((2, 3))
    for i in range(1, 4):
        b[i - 1] = points[i] - points[i - 1]
    for i in range(1, 3):
        tmp = np.cross(b[i - 1], b[i])
        n[i - 1] = tmp / np.linalg.norm(tmp)
    m = np.cross(n[0], b[1] / np.linalg.norm(b[1]))
    x = np.dot(n[0], n[1])
    y = np.dot(m, n[1])
    return -np.arctan2(y, x)


def calculate_phi_from_masked_tertiary(tertiary_masked):
    flg = 0
    phi = []
    for i, aa in enumerate(tertiary_masked):
        if flg == 0:
            flg = 1
            phi.append(0)
            continue
        # If there were missing residues
        if np.allclose(aa, np.zeros(9)):
            flg = 0
            continue
        points = np.zeros((4, 3))
        points[0] = tertiary_masked[i - 1][6:]
        points[1:] = np.reshape(aa, (3, 3))
        phi.append(calculate_dihedral_from_points(points))
    return np.array(phi) * 180 / np.pi


def calculate_psi_from_masked_tertiary(tertiary_masked):
    flg = 1
    psi = []
    sz = len(tertiary_masked)
    for i, aa in enumerate(tertiary_masked):
        if flg == 0:
            flg = 1
            continue
        # If there were missing residues
        if i + 1 == sz or np.allclose(tertiary_masked[i + 1], np.zeros(9)):
            flg = 0
            psi.append(0)
            continue
        points = np.zeros((4, 3))
        points[0:3] = np.reshape(aa, (3, 3))
        points[3] = tertiary_masked[i + 1][:3]
        psi.append(calculate_dihedral_from_points(points))
    return np.array(psi) * 180 / np.pi


def get_backbone_coords(pdb_id):
    """
    Get N, C_a and C coords of each residue and store as
    a [Length, 9] array
    """
    pdb = PDBList()
    parser = PDBParser()
    prot = pdb.retrieve_pdb_file(pdb_id, file_format="pdb", pdir="./")
    struct = parser.get_structure(pdb_id, prot)
    tertiary = []
    for model in struct:
        for chain in model:
            hlp = list(chain.get_residues())
            for j, residue in enumerate(hlp):
                if not is_aa(residue):
                    break
                if j != 0 and hlp[j - 1].get_id()[1] + 1 != residue.get_id()[1]:
                    tertiary.append(np.zeros((3, 3)))
                else:
                    tertiary.append(
                        [
                            [i for i in residue["N"].get_vector()],
                            [i for i in residue["CA"].get_vector()],
                            [i for i in residue["C"].get_vector()],
                        ]
                    )
    tertiary = np.array(tertiary)
    tertiary = np.reshape(tertiary, (-1, 9))
    return tertiary


tertiary = get_backbone_coords("1t38")
phi = calculate_phi_from_masked_tertiary(tertiary)
psi = calculate_psi_from_masked_tertiary(tertiary)
print(phi)
print(psi)
