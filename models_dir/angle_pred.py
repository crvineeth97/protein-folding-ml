import torch
import torch.nn as nn

# GENERATE INPUT

NETWORK_INPUT_RESIDUES_MAX = 2000

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
    return 0


# NETWORK


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize1, ksize2):
        super(ResidualBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=ksize1,
            padding=ksize1 // 2,
        )
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=ksize2,
            padding=ksize2 // 2,
        )

    def forward(self, input):
        residual = self.bn1(input)
        nn.functional.relu(residual, inplace=True)
        residual = self.conv1(residual)

        residual = self.bn2(residual)
        nn.functional.relu(residual, inplace=True)
        residual = self.conv2(residual)

        output = input + residual
        return output


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.blocks = []

        self.blocks.append(ResidualBlock(40, 64, 3, 3))
        for _ in range(12):
            self.blocks.append(ResidualBlock(64, 64, 3, 3))

        self.blocks.append(ResidualBlock(64, 128, 5, 5))
        for _ in range(24):
            self.blocks.append(ResidualBlock(128, 128, 5, 5))

        self.blocks.append(ResidualBlock(128, 256, 3, 3))
        for _ in range(8):
            self.blocks.append(ResidualBlock(256, 256, 3, 3))

        self.fc1 = nn.Linear(
            512 * NETWORK_INPUT_RESIDUES_MAX, 128 * NETWORK_INPUT_RESIDUES_MAX
        )
        self.fc2 = nn.Linear(
            128 * NETWORK_INPUT_RESIDUES_MAX, 64 * NETWORK_INPUT_RESIDUES_MAX
        )
        self.fc3 = nn.Linear(
            64 * NETWORK_INPUT_RESIDUES_MAX, 4 * NETWORK_INPUT_RESIDUES_MAX
        )

    def generate_input(self, pssm, primary):
        # pssm [n, 21, sequence_size]
        # primary [n, sequence_size]
        batch_size = torch.as_tensor(primary).shape[0]
        seq_len = torch.as_tensor(primary).shape[1]
        assert seq_len == NETWORK_INPUT_RESIDUES_MAX

        transformed_primary = torch.zeros(batch_size, 20, seq_len)
        for i in range(batch_size):
            for j in range(seq_len):
                residue = get_residue_index(primary[i, j])
                transformed_primary[i, residue, j] = 1

        transformed_pssm = pssm[:, 0:20, :]

        # transformed_pssm    [n, 20, L]
        # transformed_primary [n, 20, L]
        # output              [n, 40, L]
        return torch.cat((transformed_pssm, transformed_primary))

    def forward(self, input):
        output = input
        for unit in self.blocks:
            output = unit(output)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)

        return output
