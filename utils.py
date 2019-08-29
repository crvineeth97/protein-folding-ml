import numpy as np


# method for translating a point cloud to its center of mass
def transpose_atoms_to_center_of_mass(x):
    # calculate com by summing x, y and z respectively
    # and dividing by the number of points
    centerOfMass = np.array(
        [
            [x[0, :].sum() / x.shape[1]],
            [x[1, :].sum() / x.shape[1]],
            [x[2, :].sum() / x.shape[1]],
        ]
    )
    # translate points to com and return
    return x - centerOfMass


# TODO Fix calculation
def calculate_rmsd(lengths, chain_a, chain_b):
    chain_a = chain_a.cpu().numpy()
    chain_b = chain_b.cpu().numpy()
    RMSD = 0
    bs = len(lengths)
    for i in range(bs):
        a = chain_a[i, : lengths[i]].transpose()
        b = chain_b[i, : lengths[i]].transpose()
        # move to center of mass
        X = transpose_atoms_to_center_of_mass(a)
        Y = transpose_atoms_to_center_of_mass(b)

        R = np.matmul(Y, X.transpose())
        # extract the singular values
        _, S, _ = np.linalg.svd(R)
        # compute RMSD using the formula
        E0 = sum(
            list(np.linalg.norm(x) ** 2 for x in X.transpose())
            + list(np.linalg.norm(x) ** 2 for x in Y.transpose())
        )
        TraceS = sum(S)
        RMSD += np.sqrt((1 / len(X.transpose())) * (E0 - 2 * TraceS))
    RMSD /= bs
    return RMSD
