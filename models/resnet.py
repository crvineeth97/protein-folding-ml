import torch
import torch.nn as nn

from models.resnet_1d import resnet34
from parameters import MAX_PROTEIN_LENGTH

# Input Tensor
#
# PSSM [L x 20] + OH-Residue [L x 20]
# => [L x 40]

# Residue ID: [0, 20)


def get_one_hot_residue_encoding(residue):
    assert residue >= 0 and residue < 20
    output = torch.zeros(20)
    output[residue] = 1


def generate_input_tensor(pssm, residue_list):
    residue_oh_tensor = torch.zeros(len(residue_list), 20)
    for i in range(len(residue_list)):
        residue = residue_list[i]
        residue_oh_tensor[i][residue] = 1
    return torch.cat((pssm, residue_oh_tensor))


def get_residue_index(residue):
    return residue


# ######################################### NETWORK #############################################


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        # The output should be [L, 1024]
        self.reslay = resnet34()

        self.fc1 = nn.Linear(1024 * MAX_PROTEIN_LENGTH, 128 * MAX_PROTEIN_LENGTH)
        self.fc2 = nn.Linear(128 * MAX_PROTEIN_LENGTH, 4 * MAX_PROTEIN_LENGTH)

    def generate_input(self, pssm, primary, lengths):
        # pssm [n, 21, sequence_size]
        # primary [n, sequence_size]
        batch_size = len(primary)
        transformed_primary = torch.zeros(batch_size, 20, MAX_PROTEIN_LENGTH)

        for i in range(batch_size):
            assert lengths[i] <= MAX_PROTEIN_LENGTH
            for j in range(lengths[i]):
                residue = int(primary[i][j])
                transformed_primary[i][residue][j] = 1.0

        transformed_pssm = torch.zeros(batch_size, 20, MAX_PROTEIN_LENGTH)
        for i in range(batch_size):
            transformed_pssm[i] = torch.transpose(pssm[i], 0, 1)[:20, :]

        # transformed_pssm    [n, 20, L]
        # transformed_primary [n, 20, L]
        # output              [n, 40, L]
        return torch.cat((transformed_pssm, transformed_primary), dim=1)

    def forward(self, input):
        output = self.reslay(input)

        output = output.unsqueeze(dim=0)
        output = output.reshape(output.shape[1], 1, output.shape[-2], output.shape[-1])

        output = self.fc1(output.flatten(1))
        nn.functional.relu(output, inplace=True)
        output = self.fc2(output)
        output = torch.tanh(output)

        return output
