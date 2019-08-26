import torch
import torch.nn as nn


class YashasNet(nn.Module):
    def __init__(self):
        super(YashasNet, self).__init__()

        # DEPTH 1
        self.conv1 = nn.Conv1d(
            in_channels=41, out_channels=128, kernel_size=5, padding=2, bias=True
        )
        self.relu1 = nn.ReLU(inplace=True)

        # DEPTH 2
        self.conv2_1 = nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            padding=1,
            groups=2,
            bias=True,
        )
        self.relu2_1 = nn.ReLU(inplace=True)

        self.conv2_2 = nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=5,
            padding=2,
            groups=2,
            bias=True,
        )
        self.relu2_2 = nn.ReLU(inplace=True)

        self.conv2_3 = nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=7,
            padding=3,
            groups=2,
            bias=True,
        )
        self.relu2_3 = nn.ReLU(inplace=True)

        self.conv2_4 = nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=11,
            padding=5,
            groups=2,
            bias=True,
        )
        self.relu2_4 = nn.ReLU(inplace=True)

        # DEPTH 4
        self.conv3 = nn.Conv1d(
            in_channels=4 * 256,
            out_channels=1024,
            kernel_size=1,
            padding=0,
            groups=2,
            bias=True,
        )
        self.relu3 = nn.ReLU(inplace=True)

        # DEPTH 5
        self.conv4 = nn.Conv1d(
            in_channels=1024,
            out_channels=1024,
            kernel_size=3,
            padding=1,
            groups=2,
            bias=True,
        )
        self.relu4 = nn.ReLU(inplace=True)

        # DEPTH 6
        self.conv5_1 = nn.Conv1d(
            in_channels=1024,
            out_channels=2048,
            kernel_size=3,
            padding=1,
            groups=2,
            bias=True,
        )
        self.relu5_1 = nn.ReLU(inplace=True)

        self.conv5_2 = nn.Conv1d(
            in_channels=1024,
            out_channels=2048,
            kernel_size=5,
            padding=2,
            groups=2,
            bias=True,
        )
        self.relu5_2 = nn.ReLU(inplace=True)

        self.conv5_3 = nn.Conv1d(
            in_channels=1024,
            out_channels=2048,
            kernel_size=7,
            padding=3,
            groups=2,
            bias=True,
        )
        self.relu5_3 = nn.ReLU(inplace=True)

        self.conv5_4 = nn.Conv1d(
            in_channels=1024,
            out_channels=2048,
            kernel_size=11,
            padding=5,
            groups=2,
            bias=True,
        )
        self.relu5_4 = nn.ReLU(inplace=True)

        # DEPTH 7
        self.conv6 = nn.Conv1d(
            in_channels=4 * 2048, out_channels=256, kernel_size=3, padding=1, bias=True
        )
        self.relu6 = nn.ReLU(inplace=True)

        self.conv7 = nn.Conv1d(
            in_channels=256, out_channels=32, kernel_size=3, padding=1, bias=True
        )
        self.relu7 = nn.ReLU(inplace=True)

        self.conv8 = nn.Conv1d(
            in_channels=32, out_channels=4, kernel_size=3, padding=1, bias=True
        )
        self.relu8 = nn.ReLU(inplace=True)

    def forward(self, x, lengths):
        x = self.conv1(x)

        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x4 = self.conv2_4(x)
        x1 = self.relu2_1(x1)
        x2 = self.relu2_2(x2)
        x3 = self.relu2_3(x3)
        x4 = self.relu2_4(x4)
        x = torch.cat((x1, x2, x3, x4), 1)

        assert x1.numel() == x2.numel()
        assert x1.numel() == x3.numel()
        assert x1.numel() == x4.numel()
        # x = x.unsqueeze(1)
        # x = x.reshape([x.shape[0], 4, x1.shape[1], x1.shape[2]])
        # x = x.permute(0, 2, 1, 3)
        # x = x.reshape(1, x.shape[0], -1, x.shape[3])
        # x = torch.squeeze(x, dim=0)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x1 = self.conv5_1(x)
        x2 = self.conv5_2(x)
        x3 = self.conv5_3(x)
        x4 = self.conv5_4(x)
        x1 = self.relu5_1(x1)
        x2 = self.relu5_1(x2)
        x3 = self.relu5_1(x3)
        x4 = self.relu5_4(x4)
        x = torch.cat((x1, x2, x3, x4), 1)

        # assert(x1.numel() == x2.numel())
        # assert(x1.numel() == x3.numel())
        # assert(x1.numel() == x4.numel())
        # x = x.unsqueeze(1)
        # x = x.reshape([x.shape[0], 4, x1.shape[1], x1.shape[2]])
        # x = x.permute(0, 2, 1, 3)
        # x = x.reshape(1, x.shape[0], -1, x.shape[3])
        # x = torch.squeeze(x, dim=0)

        x = self.conv6(x)
        x = self.relu6(x)

        x = self.conv7(x)
        x = self.relu7(x)

        x = self.conv8(x)
        x = self.relu8(x)
        return x
