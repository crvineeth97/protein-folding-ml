import torch
from torch import nn
import torch.nn.functional as F
from constants import MINIBATCH_SIZE, DEVICE


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=41,
        n_classes=4,
        depth=5,
        wf=6,
        padding=True,
        batch_norm=False,
        up_mode="upsample",
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ("upconv", "upsample")
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv1d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x, lengths):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.avg_pool1d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return torch.tanh(self.last(x))

    def generate_input(self, lengths, primary, evolutionary):
        """
        Generate input for each minibatch. Pad the input feature vectors
        so that the final input shape is [MINIBATCH_SIZE, 41, Max_length]
        Args:
        lengths: Tuple of all protein lengths in current minibatch
        primary: Tuple of numpy arrays of shape (l,) describing the
        protein amino acid sequence, which are of variable length
        evolutionary: Tuple of numpy arrays of shape (l, 21) describing
        the PSSM matrix of the protein
        """

        transformed_primary = torch.zeros(
            MINIBATCH_SIZE, 20, lengths[0], device=DEVICE, dtype=torch.float32
        )

        # TODO: Use pythonic way
        for i in range(MINIBATCH_SIZE):
            for j in range(lengths[i]):
                residue = int(primary[i][j])
                transformed_primary[i][residue][j] = 1.0

        transformed_evolutionary = torch.zeros(
            MINIBATCH_SIZE, 21, lengths[0], device=DEVICE, dtype=torch.float32
        )
        for i in range(MINIBATCH_SIZE):
            transformed_evolutionary[i, :, : lengths[i]] = torch.from_numpy(
                evolutionary[i].T
            )

        # transformed_primary           [n, 20, L]
        # transformed_evolutionary      [n, 21, L]
        # output                        [n, 41, L]
        return torch.cat((transformed_primary, transformed_evolutionary), dim=1)

    def generate_target(self, lengths, phi, psi, omega):
        # dihedrals are in radians
        target = torch.zeros(
            MINIBATCH_SIZE, 4, lengths[0], device=DEVICE, dtype=torch.float32
        )
        for i in range(MINIBATCH_SIZE):
            ph = torch.from_numpy(phi[i])
            ps = torch.from_numpy(psi[i])
            # om = torch.from_numpy(omega[i])
            target[i, 0, : lengths[i]] = torch.sin(ph)
            target[i, 1, : lengths[i]] = torch.cos(ph)
            target[i, 2, : lengths[i]] = torch.sin(ps)
            target[i, 3, : lengths[i]] = torch.cos(ps)
            # target[i, 4, : lengths[i]] = torch.sin(om)
            # target[i, 5, : lengths[i]] = torch.cos(om)
        return target

    def calculate_loss(self, lengths, criterion, output, target):
        loss = criterion(output[0], target[0])
        for i in range(1, MINIBATCH_SIZE):
            loss += criterion(output[i, :, : lengths[i]], target[i, :, : lengths[i]])
        loss /= MINIBATCH_SIZE
        return loss


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv1d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.Tanh())
        if batch_norm:
            block.append(nn.BatchNorm1d(out_size))

        block.append(nn.Conv1d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.Tanh())
        if batch_norm:
            block.append(nn.BatchNorm1d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == "upconv":
            self.up = nn.ConvTranspose1d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == "upsample":
            self.up = nn.Sequential(
                nn.Upsample(mode="linear", scale_factor=2),
                nn.Conv1d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        # diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y : (diff_y + target_size[0])]

    def pad(self, up, dim_2):
        if dim_2 == 0:
            return up
        return torch.cat(
            [up, torch.zeros(MINIBATCH_SIZE, up.shape[1], dim_2).to(DEVICE)], 2
        )

    def forward(self, x, bridge):
        up = self.up(x)
        # crop1 = self.center_crop(bridge, up.shape[2:])
        up = self.pad(up, bridge.shape[2] - up.shape[2])
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)

        return out
