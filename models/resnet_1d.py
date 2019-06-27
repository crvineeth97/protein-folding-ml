import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=[3, 3],
        groups=1,
        base_width=64,
        norm_layer=None,
    ):
        # Since we need same length output, we can't have downsampling
        # or dilations or strides
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if type(kernel_size) is not list or len(kernel_size) != 2:
            raise ValueError("BasicBlock requires a list of length 2 for kernel_size")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
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

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        groups=1,
        base_width=64,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if type(kernel_size) is not int:
            raise ValueError("BottleneckBlock requires an integer for kernel_size")
        width = int(out_channels * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv1d(
            in_channels=in_channels, out_channels=width, kernel_size=1, bias=False
        )
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv1d(
            in_channels=width,
            out_channels=width,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
            bias=False,
        )
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv1d(
            in_channels=width,
            out_channels=out_channels * self.expansion,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = norm_layer(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class MakeResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        kernel_size,
        feat_vec_len=41,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        norm_layer=None,
    ):
        super(MakeResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv1d(
            in_channels=feat_vec_len,
            out_channels=self.inplanes,
            kernel_size=7,
            padding=3,
            bias=False,
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], kernel_size)
        self.layer2 = self._make_layer(block, 128, layers[1], kernel_size)
        self.layer3 = self._make_layer(block, 256, layers[2], kernel_size)
        self.layer4 = self._make_layer(block, 512, layers[3], kernel_size)

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
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, kernel_size):
        norm_layer = self._norm_layer

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                kernel_size,
                self.groups,
                self.base_width,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    kernel_size,
                    self.groups,
                    self.base_width,
                    norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    return MakeResNet(BasicBlock, [2, 2, 2, 2], [3, 3], **kwargs)


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    return MakeResNet(BasicBlock, [3, 4, 6, 3], [3, 3], **kwargs)


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    return MakeResNet(Bottleneck, [3, 4, 6, 3], 3, **kwargs)


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    return MakeResNet(Bottleneck, [3, 4, 23, 3], 3, **kwargs)


def resnet152(**kwargs):
    """Constructs a ResNet-152 model.
    """
    return MakeResNet(Bottleneck, [3, 8, 36, 3], 3, **kwargs)


def resnext50_32x4d(**kwargs):
    """Constructs a ResNeXt-50 32x4d model.
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return MakeResNet(Bottleneck, [3, 4, 6, 3], 3, **kwargs)


def resnext101_32x8d(**kwargs):
    """Constructs a ResNeXt-101 32x8d model.
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return MakeResNet(Bottleneck, [3, 4, 23, 3], 3, **kwargs)


def wide_resnet50_2(**kwargs):
    """Constructs a Wide ResNet-50-2 model.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    kwargs["width_per_group"] = 64 * 2
    return MakeResNet(Bottleneck, [3, 4, 6, 3], 3, **kwargs)


def wide_resnet101_2(**kwargs):
    """Constructs a Wide ResNet-101-2 model.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    kwargs["width_per_group"] = 64 * 2
    return MakeResNet(Bottleneck, [3, 4, 23, 3], 3, **kwargs)
