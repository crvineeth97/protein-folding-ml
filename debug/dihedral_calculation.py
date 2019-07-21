import sys

import numpy as np
import PeptideBuilder
from Bio.PDB import PDBList, PDBParser, is_aa, PDBIO
from Bio.PDB.Polypeptide import three_to_one

sys.path.append("../")
from pnerf.pnerf import dihedral_to_point


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
    """
    The phi angle of an amino acid at the i'th position is calculated
    using the coordinates of the C{i-1}, N{i}, Ca{i}, C{i} atoms. Hence,
    the phi angle of the first amino acid is always 0
    """
    flg = 0
    phi = []
    for i, aa in enumerate(tertiary_masked):
        # If there were missing residues, we don't have their tertiary
        # positions and hence phi will be 0 for the first amino acid
        # after the missing residues, because we don't have C{i-1}
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
    return np.array(phi, dtype=np.float) * 180.0 / np.pi


def calculate_psi_from_masked_tertiary(tertiary_masked):
    """
    The psi angle of an amino acid at the i'th position is calculated
    using the coordinates of the N{i}, Ca{i}, C{i}, N{i+1} atoms. Hence,
    the psi angle of the last amino acid is always 0
    """
    flg = 1
    psi = []
    sz = len(tertiary_masked)
    for i, aa in enumerate(tertiary_masked):
        # If there were missing residues, we don't have their tertiary
        # positions and hence psi will be 0 for the last amino acid
        # before the missing residues, because we don't have N{i+1}
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
    return np.array(psi, dtype=np.float) * 180.0 / np.pi


def calculate_omega_from_masked_tertiary(tertiary_masked):
    """
    The omega angle of an amino acid at the i'th position is calculated
    using the coordinates of the Ca{i}, C{i}, N{i+1}, Ca{i+1} atoms. Hence,
    the omega angle of the last amino acid is always 0
    """
    flg = 1
    omega = []
    sz = len(tertiary_masked)
    for i, aa in enumerate(tertiary_masked):
        # If there were missing residues, we don't have their tertiary
        # positions and hence omega will be 0 for the last amino acid
        # before the missing residues, because we don't have C{i-1}
        if i + 1 == sz or np.allclose(tertiary_masked[i + 1], np.zeros(9)):
            flg = 0
            omega.append(0)
            continue
        if flg == 0:
            flg = 1
            continue
        points = np.zeros((4, 3))
        points[0:2] = np.reshape(aa[3:], (2, 3))
        points[2:] = np.reshape(tertiary_masked[i + 1][:6], (2, 3))
        omega.append(calculate_dihedral_from_points(points))
    return np.array(omega, dtype=np.float) * 180.0 / np.pi


def get_backbone_coords(pdb_id):
    """
    Get N, C_a and C coords of each residue and store as
    a [Length, 9] array
    """
    pdb = PDBList()
    parser = PDBParser()
    prot = pdb.retrieve_pdb_file(pdb_id, file_format="pdb", pdir="./")
    struct = parser.get_structure(pdb_id, prot)
    primary = []
    tertiary = []
    for model in struct:
        for chain in model:
            hlp = list(chain.get_residues())
            for j, residue in enumerate(hlp):
                if not is_aa(residue):
                    break
                if j != 0 and hlp[j - 1].get_id()[1] + 1 != residue.get_id()[1]:
                    tertiary.append(np.zeros((3, 3)))
                primary.append(three_to_one(residue.get_resname()))
                tertiary.append(
                    [
                        [i for i in residue["N"].get_vector()],
                        [i for i in residue["CA"].get_vector()],
                        [i for i in residue["C"].get_vector()],
                    ]
                )
    tertiary = np.array(tertiary)
    tertiary = np.reshape(tertiary, (-1, 9))
    return primary, tertiary


primary, tertiary = get_backbone_coords("2YO0")
phi = calculate_phi_from_masked_tertiary(tertiary)
psi = calculate_psi_from_masked_tertiary(tertiary)
omega = calculate_omega_from_masked_tertiary(tertiary)
print(primary)
print(phi)
print()
print(psi)
print()
print(omega)
print()

# dihedrals = from_numpy(
#     np.reshape(np.array(list(zip(phi, psi, omega)), dtype=np.float32), (-1, 1, 3))
# )
# points = dihedral_to_point(dihedrals, device("cpu"))

# print(points)
structure = PeptideBuilder.make_structure(primary, phi, psi)
out = PDBIO()
out.set_structure(structure)
out.save("test.pdb")
