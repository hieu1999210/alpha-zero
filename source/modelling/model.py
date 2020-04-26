import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        base_channel = 128
        H = 6
        W = 6
        bn_channels = 32
        input_channels = 1
        num_actions = 37
        self.backbone = nn.Sequential(
            conv3x3(input_channels, base_channel),
            nn.BatchNorm2d(base_channel),
            *[Bottleneck(base_channel, bn_channels, base_channel)]*19
        )
        self.policy_head = PolicyHead(base_channel, H,W, num_actions)
        self.value_head = ValueHead(base_channel, H, W)
    
    def forward(self, x):
        features = self.backbone(x)
        value = self.value_head(features)
        policy = self.policy_head(features)
        
        # return value, policy
        return policy



class PolicyHead(nn.Module):
    def __init__(self, in_channels, H,W, num_actions):
        super(PolicyHead, self).__init__()
        self.layers = nn.Sequential(
            conv1x1(in_channels, 2),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2*H*W, num_actions)
        )
    
    def forward(self, x):
        return self.layers(x)


class ValueHead(nn.Module):
    def __init__(self, in_channels, H,W):
        super(ValueHead, self).__init__()
        self.layers = nn.Sequential(
            conv1x1(in_channels, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(H*W, 256),
            nn.ReLU(),
            nn.Linear(256,1),
        )
    
    def forward(self, x):
        x = self.layers(x)
        return torch.tanh(x)


class Bottleneck(nn.Module):
    def __init__(
        self, in_channels, bottleneck_channels, out_channels,
        stride=1, downsample=None, groups=1,
        norm_layer=None, drop_rate=0.0, block_drop=0.0
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_channels, bottleneck_channels)
        self.bn1 = norm_layer(bottleneck_channels)
        self.conv2 = conv3x3(bottleneck_channels, bottleneck_channels, stride)
        self.bn2 = norm_layer(bottleneck_channels)
        self.conv3 = conv1x1(bottleneck_channels, out_channels)
        self.bn3 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout3d(drop_rate, inplace=False)
        self.block_drop = block_drop
        
    def forward(self, x):
        identity = x

        out=0
        if (self.training and torch.rand(1) > self.block_drop) or not self.training:
            
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.dropout(out)

            out = self.conv3(out)
            out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.training:
            out += identity
        else:
            out = out*(1-self.block_drop) + identity
            
        out = self.relu(out)

        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )
    
    
if __name__ == "__main__":
    model = Model(None)
    a = torch.randn((2,1,8,8))
    value, policy = model(a)
    print(model)
    print(value.shape, policy.shape)