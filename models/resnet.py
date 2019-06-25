import torch
import torch.nn as nn

# There are two networks: 1d and 2d
# Block1d represents the residual block of the 1d network
# Network1d represents the 1d network

SEQUENCE_LENGTH = 30

# ######################################### NETWORK 1D ###########################################

NETWORK1_INPUT_FEATURES = 26
NETWORK1_INPUT_SIZE = (1, NETWORK1_INPUT_FEATURES, SEQUENCE_LENGTH)

NETWORK1_RESIDUAL_BLOCK_CONV1_KSIZE = 17
NETWORK1_RESIDUAL_BLOCK_CONV2_KSIZE = 17


class Block1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block1d, self).__init__()
        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=NETWORK1_RESIDUAL_BLOCK_CONV1_KSIZE,
            padding=NETWORK1_RESIDUAL_BLOCK_CONV1_KSIZE // 2,
        )
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=NETWORK1_RESIDUAL_BLOCK_CONV2_KSIZE,
            padding=NETWORK1_RESIDUAL_BLOCK_CONV1_KSIZE // 2,
        )

    def forward(self, input):
        residual = input

        residual = self.bn1(residual)
        nn.functional.relu(residual, inplace=True)
        residual = self.conv1(residual)

        residual = self.bn2(residual)
        nn.functional.relu(residual, inplace=True)
        residual = self.conv2(residual)

        # input shape = (1, in_channels, SEQUENCE_LENGTH)
        # residual shape = (1, filters, SEQUENCE_LENGTH)
        # filters >= in_channels

        in_channels = input.shape[1]
        padded_input = torch.zeros(residual.shape)
        padded_input[:, :in_channels, :] = input

        output = padded_input + residual
        return output


class Network1d(nn.Module):
    def __init__(self,):
        super(Network1d, self).__init__()
        self.blocks = []
        self.blocks.append(Block1d(NETWORK1_INPUT_FEATURES, 36))
        self.blocks.append(Block1d(36, 72))
        self.blocks.append(Block1d(72, 144))

    # input = L x n
    def forward(self, input):
        output = input
        for unit in self.blocks:
            output = unit(output)
        return output


# ######################################### NETWORK 2D ###########################################

NETWORK2_INPUT_FEATURES = 432
NETWORK2_INPUT_SIZE = (1, NETWORK1_INPUT_FEATURES, SEQUENCE_LENGTH, SEQUENCE_LENGTH)

NETWORK2_RESIDUAL_BLOCK_CONV1_KSIZE = 3
NETWORK2_RESIDUAL_BLOCK_CONV2_KSIZE = 5


class Block2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block2d, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=NETWORK2_RESIDUAL_BLOCK_CONV1_KSIZE,
            padding=NETWORK2_RESIDUAL_BLOCK_CONV1_KSIZE // 2,
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=NETWORK2_RESIDUAL_BLOCK_CONV2_KSIZE,
            padding=NETWORK2_RESIDUAL_BLOCK_CONV2_KSIZE // 2,
        )

    def forward(self, input):
        residual = input

        residual = self.bn1(residual)
        nn.functional.relu(residual, inplace=True)
        residual = self.conv1(residual)

        residual = self.bn2(residual)
        nn.functional.relu(residual, inplace=True)
        residual = self.conv2(residual)

        # input shape = (1, in_channels, SEQUENCE_LENGTH, SEQUENCE_LENGTH)
        # residual shape = (1, filters, SEQUENCE_LENGTH, SEQUENCE_LENGTH)
        # filters <= in_channels

        out_channels = residual.shape[1]
        padded_residual = torch.zeros(input.shape)
        padded_residual[:, :out_channels, :, :] = residual

        output = input + padded_residual
        return output


class Network2d(nn.Module):
    def __init__(self,):
        super(Network2d, self).__init__()
        self.blocks = []
        self.blocks.append(Block2d(NETWORK2_INPUT_FEATURES, 60))
        for _ in range(29):
            self.blocks.append(Block2d(60, 60))

    # input = L x n
    def forward(self, input):
        output = input
        for unit in self.blocks:
            output = unit(output)
        return output


# ########################## SEQUENTIAL TO PAIRWISE FEATURE CONVERSION ############################

# input shape: [1, n, L]
# output shape: [1, m, L, L]
def convert_seq2pair(input_tensor):
    residues = input_tensor.shape[-1]
    features = input_tensor.shape[-2]
    output_tensor = torch.zeros((1, features * 3, residues, residues))

    for i in range(residues):
        vi = input_tensor[0][:, i : i + 1]
        for j in range(residues):
            vj = input_tensor[0][:, j : j + 1]
            ij2 = (i + j) // 2
            vij2 = input_tensor[0][:, ij2 : ij2 + 1]
            output_tensor[0][:, i, j] = torch.cat((vi, vij2, vj)).reshape(-1).squeeze()

    return output_tensor


# ########################## CO-EVOLUTION AND PAIRWISE POTENTIAL INFO ############################


def concat_extra_info(input_tensor, extra_tensor):
    # extra_tensor shape [1, 3, L, L]
    # input_tensor shape [1, m, L, L]
    return torch.cat((input_tensor, extra_tensor), dim=1)


# ###################################### GENERATE INPUT ##########################################


def generate_input_tensor(pssm, ss3_pred, sa_pred):
    # pssm [1, 20, L]
    # - we take this from ProteinNet
    #
    # predicted secondary structure [1, 3, L]; features are probabilities/scores for the structure
    # - ??
    #
    # predicted solvent accessiblity [1, 3, L]; features are scores for 3 states (buried, intermeddiate, exposed)
    # - ??
    return torch.cat((pssm, ss3_pred, sa_pred))


# ############################################ MAIN ##############################################


def main():
    input1d = torch.zeros((*NETWORK1_INPUT_SIZE), dtype=torch.float32)
    net1d = Network1d()
    output1d = net1d.forward(input1d)
    print(input1d.shape)

    print(output1d.shape)
    input2d = convert_seq2pair(output1d)
    print(input2d.shape)
    net2d = Network2d()
    output2d = net2d.forward(input2d)

    print(output2d)


main()
