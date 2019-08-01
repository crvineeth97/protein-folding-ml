import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=[17, 17],
        norm_layer=None,
        downsample=None,
    ):
        # Since we need same length output, we can't have
        # downsampling, dilations or strides
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if type(kernel_size) is not list or len(kernel_size) != 2:
            raise ValueError("BasicBlock requires a list of length 2 for kernel_size")
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size[0],
            padding=kernel_size[0] // 2,
            bias=False,
        )
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size[1],
            padding=kernel_size[1] // 2,
            bias=False,
        )
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out += identity
        out = self.relu(out)

        return out


class MakeResNet(nn.Module):
    def __init__(
        self,
        layers,
        kernel_size,
        feat_vec_len=41,
        zero_init_residual=False,
        norm_layer=None,
    ):
        super(MakeResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.start_planes = 64
        self.conv1 = nn.Conv1d(
            in_channels=feat_vec_len,
            out_channels=self.start_planes,
            kernel_size=7,
            padding=3,
            bias=False,
        )
        self.bn1 = norm_layer(self.start_planes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(64, layers[0], kernel_size)
        self.layer2 = self._make_layer(128, layers[1], kernel_size)
        self.layer3 = self._make_layer(256, layers[2], kernel_size)
        self.layer4 = self._make_layer(512, layers[3], kernel_size)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each
        # residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, out_planes, blocks, kernel_size):
        norm_layer = self._norm_layer
        downsample = nn.Sequential(
            nn.Conv1d(
                in_channels=self.start_planes,
                out_channels=out_planes,
                kernel_size=1,
                bias=False,
            ),
            norm_layer(out_planes),
        )
        layers = []
        layers.append(
            BasicBlock(
                self.start_planes, out_planes, kernel_size, norm_layer, downsample
            )
        )
        self.start_planes = out_planes
        for _ in range(1, blocks):
            layers.append(
                BasicBlock(self.start_planes, out_planes, kernel_size, norm_layer)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    return MakeResNet([2, 2, 2, 2], [17, 17], **kwargs)


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    return MakeResNet([3, 4, 6, 3], [17, 17], **kwargs)
