import torch
import torch.nn as nn
import random

######################################## GENERATE INPUT ##########################################

NETWORK_INPUT_RESIDUES_MAX = 1000

# Input Tensor
#
# PSSM [L x 20] + OH-Residue [L x 20]
# => [L x 40]

# Residue ID: [0, 20)

def get_one_hot_residue_encoding(residue):
    assert(residue >= 0 and residue < 20)
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

########################################### NETWORK #############################################

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize1, ksize2):
        super(ResidualBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=ksize1,
                               padding=ksize1//2)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=ksize2,
                               padding=ksize2//2)

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
    def __init__(self, device):
        super(Network, self).__init__()
        self.blocks = []
        
        self.blocks.append(ResidualBlock(in_channels=40,
                                         out_channels=40,
                                         ksize1=3, ksize2=3).to(device))
        for _ in range(64):
            self.blocks.append(ResidualBlock(40, 40, 3, 3).to(device))

        self.fc1 = nn.Linear(40 * NETWORK_INPUT_RESIDUES_MAX, 16 * NETWORK_INPUT_RESIDUES_MAX)
        self.fc2 = nn.Linear(16 * NETWORK_INPUT_RESIDUES_MAX,  4 * NETWORK_INPUT_RESIDUES_MAX)

    def generate_input(self, pssm, primary, lengths):
        # pssm [n, 21, sequence_size]
        # primary [n, sequence_size]
        batch_size = len(primary)
        transformed_primary = torch.zeros(batch_size, 20, NETWORK_INPUT_RESIDUES_MAX)

        for i in range(batch_size):
            assert(lengths[i] <= NETWORK_INPUT_RESIDUES_MAX)
            for j in range(lengths[i]):
                residue = int(primary[i][j])
                transformed_primary[i][residue][j] = 1.0

        transformed_pssm = torch.zeros(batch_size, 20, NETWORK_INPUT_RESIDUES_MAX)
        for i in range(batch_size):
            transformed_pssm[i] = torch.transpose(pssm[i], 0, 1)[:20, :]

        # transformed_pssm    [n, 20, L]
        # transformed_primary [n, 20, L]
        # output              [n, 40, L]
        return torch.cat((transformed_pssm, transformed_primary), dim=1)

    def forward(self, input):
        output = input
        for unit in self.blocks:
            output = unit(output)

        output = output.unsqueeze(dim=0)
        output = output.reshape(output.shape[1], 1, output.shape[-2], output.shape[-1])

        output = self.fc1(output.flatten(1))
        nn.functional.relu(output, inplace=True)
        output = self.fc2(output)
        output = torch.tanh(output)

        return output

