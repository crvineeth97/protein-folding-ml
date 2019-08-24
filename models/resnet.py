import torch.nn as nn

from models.resnet_1d import resnet6


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.reslay = resnet6()
        self.fc1 = nn.Linear(512, 64)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(64, 4)
        self.act2 = nn.Tanh()

    def forward(self, x, lengths):
        # [Batch, 41, Max_length] -> [Batch, 512, Max_length]
        output = self.reslay(x)

        # [Batch, 512, Max_length] -> [Batch * Max_length, 512]
        output = output.transpose(1, 2)
        output = output.contiguous().view(-1, output.shape[2])

        # Run through linear and activation layers
        output = self.fc1(output)
        output = self.act1(output)
        output = self.fc2(output)
        output = self.act2(output)

        # [Batch * Max_length, 4] -> [Batch, 4, Max_length]
        output = output.view(-1, lengths[0], 4)
        output = output.transpose(1, 2)

        return output
