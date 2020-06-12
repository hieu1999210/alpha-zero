import torch.nn as nn
import torch
import torch.nn.functional as F

def pi_loss(logits, targets):
    
    log_prob = torch.log_softmax(logits, dim=1)
    return -(log_prob*targets).sum(dim=1).mean()

class BaseModel(nn.Module):
    
    def freeze_param(self):
        for param in self.parameters():
            param.requires_grad = False
        self.is_freezed = True
        
    def un_freeze(self):
        for param in self.parameters():
            param.requires_grad = True
        self.is_freezed = False


class Resnet(BaseModel):
    def __init__(self, game, cfg):
        super(Resnet, self).__init__()
        
        base_channel    = cfg.MODEL.BASE_CHANNELS
        H, W            = game.getBoardSize()
        bn_channels     = cfg.MODEL.BOTTLENECK_CHANNELS
        input_channels  = 1
        num_actions     = game.getActionSize()
        num_blocks      = cfg.MODEL.NUM_BLOCKS
        drop_rate       = cfg.MODEL.DROP_RATE
        
        self.backbone = nn.Sequential(
            conv3x3(input_channels, base_channel),
            nn.BatchNorm2d(base_channel),
            *[Bottleneck(
                in_channels=base_channel, 
                bottleneck_channels=bn_channels, 
                out_channels=base_channel,
                drop_rate=drop_rate,
                block_drop=cfg.MODEL.BLOCK_DROP,
            )]*num_blocks
        )
        self.policy_head = PolicyHead(
            base_channel, H, W, num_actions, drop_rate)
        self.value_head = ValueHead(base_channel, H, W, drop_rate)
        self.is_freezed = False
        
        self.value_loss = nn.MSELoss()
        self.policy_loss = pi_loss
    
    def forward(self, x):
        features = self.backbone(x)
        value = self.value_head(features)
        policy = self.policy_head(features)
        
        if not self.training:
            policy = torch.softmax(policy, dim=1)
        return policy, value
    
    def loss(self, preds, gts):
        p_loss = self.policy_loss(preds[0], gts[0])
        v_loss = self.value_loss(preds[1], gts[1])
        
        return {
            "pi_loss": p_loss,
            "v_loss": v_loss,
        }
        # return p_loss

"""
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
"""


class OthelloNet(BaseModel):
    
    def __init__(self, game):
        super(OthelloNet, self).__init__()
        
        num_channels = 512
        self.num_channels = num_channels
        self.dropout = 0.3
        self.is_freezed = False
        self.value_loss = nn.MSELoss()
        self.policy_loss = pi_loss
        
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        self.conv1 = nn.Conv2d(1,  num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_channels,  num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d( num_channels,  num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d( num_channels,  num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d( num_channels)
        self.bn2 = nn.BatchNorm2d( num_channels)
        self.bn3 = nn.BatchNorm2d( num_channels)
        self.bn4 = nn.BatchNorm2d( num_channels)

        self.fc1 = nn.Linear( num_channels*(self.board_x-4)*(self.board_y-4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, self.num_channels*(self.board_x-4)*(self.board_y-4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1
        if not self.training:
            pi = torch.softmax(pi, dim=1)
            
        return pi, torch.tanh(v)

    def loss(self, preds, gts):
        p_loss = self.policy_loss(preds[0], gts[0])
        v_loss = self.value_loss(preds[1], gts[1])
        
        return {
            "pi_loss": p_loss,
            "v_loss": v_loss,
        }


class PolicyHead(nn.Module):
    def __init__(self, in_channels, H, W, num_actions, drop_rate=.0):
        super(PolicyHead, self).__init__()
        self.layers = nn.Sequential(
            conv1x1(in_channels, 2),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(p=drop_rate),
            nn.Linear(2*H*W, num_actions)
        )
    
    def forward(self, x):
        return self.layers(x)


class ValueHead(nn.Module):
    def __init__(self, in_channels, H, W, drop_rate=.0):
        super(ValueHead, self).__init__()
        self.layers = nn.Sequential(
            conv1x1(in_channels, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(p=drop_rate),
            nn.Linear(H*W, 256),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
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
