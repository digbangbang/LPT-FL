import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision.models.resnet import ResNet, BasicBlock

device = 'cuda' if torch.cuda.is_available() else 'cpu'

channel_dict = {
    "cifar10": 3,
    "cinic10": 3,
    "cifar100": 3,
    "mnist": 1,
    "fmnist": 1,
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, quant=None, size=32):
        super().__init__()

        self.quant = quant

        self.res_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.res_conv2 = nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1,
                                   bias=False)
        self.res_norm1 = nn.BatchNorm2d(out_channels)
        self.res_norm2 = nn.BatchNorm2d(out_channels * BasicBlock.expansion)

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.use_shortcut = True
        else:
            self.use_shortcut = False

        if self.use_shortcut:
            self.shortcut_conv1 = nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1,
                                            stride=stride, bias=False)
            self.shortcut_norm1 = nn.BatchNorm2d(out_channels * BasicBlock.expansion)

    def res_func(self, x):
        out = F.relu(self.res_norm1(self.res_conv1(x)))
        if self.quant is not None:
            out = self.quant(out)
        out = self.res_norm2(self.res_conv2(out))
        if self.quant is not None:
            out = self.quant(out)
        return out

    def shortcut(self, x):
        if self.use_shortcut:
            out = self.shortcut_norm1(self.shortcut_conv1(x))
            if self.quant is not None:
                out = self.quant(out)
            return out
        else:
            return 0.

    def forward(self, x):
        out =  F.relu((self.res_func(x) + self.shortcut(x)))
        if self.quant is not None:
            out = self.quant(out)
        return out


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, quant=None, size=32):
        super().__init__()
        self.quant = quant
        self.res_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.res_norm1 = nn.BatchNorm2d(out_channels)
        self.res_conv2 = nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False)
        self.res_norm2 = nn.BatchNorm2d(out_channels)
        self.res_conv3 = nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False)
        self.res_norm3 = nn.BatchNorm2d(out_channels * BottleNeck.expansion)

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.use_shortcut = True
        else:
            self.use_shortcut = False
        if self.use_shortcut:
            self.shortcut_conv1 = nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False)
            self.shortcut_norm1 = nn.BatchNorm2d(out_channels * BottleNeck.expansion)

    def res_func(self, x):
        out = F.relu(self.res_norm1(self.res_conv1(x)))
        if self.quant is not None:
            out = self.quant(out)
        out = F.relu(self.res_norm2(self.res_conv2(out)))
        if self.quant is not None:
            out = self.quant(out)
        out = self.res_norm3(self.res_conv3(out))
        if self.quant is not None:
            out = self.quant(out)
        return out

    def shortcut(self, x):
        if self.use_shortcut:
            out = self.shortcut_norm1(self.shortcut_conv1(x))
            if self.quant is not None:
                out = self.quant(out)
            return out
        else:
            return 0.

    def forward(self, x):
        out = F.relu((self.res_func(x) + self.shortcut(x)))
        if self.quant is not None:
            out = self.quant(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100, dataset='cifar10', quant=None):
        super().__init__()

        self.in_channels = 64
        self.quant = quant()
        channel = channel_dict.get(dataset)
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1, size=32)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2, size=32)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2, size=16)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2, size=8)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride, size):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, size=size, quant=self.quant))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        if self.quant is not None:
            output = self.quant(output)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        if self.quant is not None:
            output = self.quant(output)

        return output


def resnet18(dataset, num_classes, num_bloak=[2, 2, 2, 2], quant=None):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, num_block=num_bloak, num_classes=num_classes, dataset=dataset, quant=quant)


def resnet8(dataset, num_classes, num_block=[1, 1, 1, 1], quant=None):
    return ResNet(BasicBlock, num_block=num_block, num_classes=num_classes, dataset=dataset, quant=quant)


class MLP(nn.Module):
    def __init__(self, num_classes=10, net_width=128, im_size=(28, 28), dataset='cifar10', quant=None):
        super(MLP, self).__init__()
        channel = channel_dict.get(dataset)
        self.quant_layer = quant()
        self.base = self._make_layers(im_size, channel, net_width)
        self.classifier = nn.Linear(net_width, num_classes)

    def _make_layers(self, im_size, channel, net_width):
        layers = []
        layers += [nn.Flatten()]
        layers += [nn.Linear(im_size[0] * im_size[1] * channel, net_width)]
        layers += [nn.ReLU()]
        layers += [self.quant_layer]
        layers += [nn.Linear(net_width, net_width)]
        layers += [nn.ReLU()]
        layers += [self.quant_layer]

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.base(x)
        x = self.classifier(x)
        x = self.quant_layer(x)
        return x


class ConvNet(nn.Module):
    def __init__(self, num_classes=10, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm',
                 net_pooling='avgpooling', im_size=(32, 32), dataset='cifar10', quant=None):
        super(ConvNet, self).__init__()
        channel = channel_dict.get(dataset)
        self.quant = quant()
        self.base, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling,
                                                      im_size)
        num_feat = shape_feat[0] * shape_feat[1] * shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)
        self.quant_layer = self.quant

    def forward(self, x):
        out = self.base(x)
        out = self.classifier(out)
        out = self.quant_layer(out)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s' % net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s' % net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s' % net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            layers += [self.quant]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            layers += [self.quant]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2
                layers += [self.quant]
        layers += [nn.Flatten()]

        return nn.Sequential(*layers), shape_feat


class CGeneratorA(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1, img_size=32, n_cls=10):
        super(CGeneratorA, self).__init__()

        self.init_size = img_size//4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf*self.init_size**2))
        self.l2 = nn.Sequential(nn.Linear(n_cls, ngf*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(ngf*2),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(nc, affine=False)
        )

    def forward(self, z, y):
        out_1 = self.l1(z.view(z.shape[0],-1))
        out_2 = self.l2(y.view(y.shape[0],-1))
        out = torch.cat([out_1, out_2], dim=1)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img


def get_model(model):
    return {
        "resnet8": (resnet8, optim.Adam, {"lr": 0.001}),
        "resnet18": (resnet18, optim.Adam, {"lr": 0.001}),
        "ConvNet": (ConvNet, optim.Adam, {"lr": 0.001}),
        "MLP": (MLP, optim.Adam, {"lr": 0.001}),
        'Generator': CGeneratorA,
    }[model]
