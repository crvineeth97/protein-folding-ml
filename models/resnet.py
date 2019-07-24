import torch
import torch.nn as nn

from models.resnet_1d import resnet34
from numpy import pi
from constants import MINIBATCH_SIZE, DEVICE


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.reslay = resnet34()
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 6)

    def generate_input(self, lengths, primary, evolutionary):
        """
        Generate input for each minibatch. Pad the input feature vectors
        so that the final input shape is [MINIBATCH_SIZE, Max_length, 41]
        Args:
        lengths: Tuple of all protein lengths in current minibatch
        primary: Tuple of numpy arrays of shape (l,) describing the
        protein amino acid sequence, which are of variable length
        evolutionary: Tuple of numpy arrays of shape (l, 21) describing
        the PSSM matrix of the protein
        """

        transformed_primary = torch.zeros(MINIBATCH_SIZE, 20, lengths[0], device=DEVICE)

        for i in range(MINIBATCH_SIZE):
            for j in range(lengths[i]):
                residue = int(primary[i][j])
                transformed_primary[i][residue][j] = 1.0

        transformed_evolutionary = torch.zeros(
            MINIBATCH_SIZE, 21, lengths[0], device=DEVICE
        )
        for i in range(MINIBATCH_SIZE):
            transformed_evolutionary[i, :, : lengths[i]] = torch.transpose(
                torch.from_numpy(evolutionary[i]), 0, 1
            )

        # transformed_primary           [n, 20, L]
        # transformed_evolutionary      [n, 21, L]
        # output                        [n, 41, L]
        return torch.cat((transformed_primary, transformed_evolutionary), dim=1)

    def generate_target(self, lengths, phi, psi, omega):
        # dihedrals are in degrees
        target = torch.zeros(MINIBATCH_SIZE, 6, lengths[0], device=DEVICE)
        for i in range(MINIBATCH_SIZE):
            ph = torch.from_numpy(phi[i] * pi / 180.0)
            ps = torch.from_numpy(psi[i] * pi / 180.0)
            om = torch.from_numpy(omega[i] * pi / 180.0)
            target[i, 0, : lengths[i]] = torch.sin(ph)
            target[i, 1, : lengths[i]] = torch.cos(ph)
            target[i, 2, : lengths[i]] = torch.sin(ps)
            target[i, 3, : lengths[i]] = torch.sin(ps)
            target[i, 4, : lengths[i]] = torch.sin(om)
            target[i, 5, : lengths[i]] = torch.sin(om)
        return target

    def calculate_loss(self, lengths, criterion, output, target):
        loss = criterion(output[0], target[0])
        for i in range(1, MINIBATCH_SIZE):
            loss += criterion(output[i, :, : lengths[i]], target[i, :, : lengths[i]])
        loss /= MINIBATCH_SIZE
        return loss

    def forward(self, input):
        # [Batch, 41, Max_length]
        output = self.reslay(input)
        # [Batch, 512, Max_length]
        output = output.transpose(1, 2)
        # [Batch, Max_length, 512]
        output = self.fc1(output)
        output = nn.functional.relu(output)
        # [Batch, Max_length, 64)]
        output = self.fc2(output)
        output = torch.tanh(output)
        # [Batch, Max_length, 4)]
        output = output.transpose(1, 2)
        # [Batch, 4, Max_length]
        return output
