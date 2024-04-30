import torch.nn.functional as F
import torch.nn as nn
from .upsampling import UpSamplingBilinear_me
import math
import torch


class eca_attention(nn.Module):
    def __init__(self, k_size=3):
        super(eca_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = F.sigmoid(y)
        return x * y.expand_as(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class spatial_attention(nn.Module):
    def __init__(self, use_cbam=False):
        super(spatial_attention, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
        self.use_cbam = False

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class BatchNormalization(nn.Module):
    def __init__(self, batch_size=32):
        super(BatchNormalization, self).__init__()
        self.batch_size = batch_size

    def forward(self, x):
        batch = nn.BatchNorm2d(self.batch_size).to(torch.device("cuda"))
        return batch(x)


class get_input_size(nn.Module):
    def __init__(self):
        super(get_input_size, self).__init__()

    def forward(self, x):
        batch = x.shape[0]
        channel = x.shape[1]
        height = x.shape[2]
        return batch, channel, height

class PAB(nn.Module):
    print("using pab")
    def __init__(self, get_channels, ratio=2, re_co=0.01, channeul_num=2, link_place=0):
        super(PAB, self).__init__()
        self.get_channels = get_channels
        self.ratio = ratio
        self.re_co = re_co
        self.channel_num = channeul_num
        self.link_place = link_place
        self.get_input_size = get_input_size()
        self.convway = convway(get_channels, re_co=self.re_co, channel_num=self.channel_num)
        self.eca = eca_attention()
        self.batchnormal = nn.BatchNorm2d(25)
        self.spatial = spatial_attention()

    def forward(self, x_input):
        # print(input.shape)
        batch, channel, height = self.get_input_size(x_input)
        low_level_feature, high_level_feature = self.convway(x_input)

        high_level_feature = x_input + high_level_feature
        x = self.eca(high_level_feature)
        x = F.sigmoid(x)
        batch = x_input.size()[1]
        x = BatchNormalization(batch)(x).to(torch.device("cuda"))
        x = UpSamplingBilinear_me(targetsize=height)(x).to(torch.device("cuda"))
        x1 = x
        x = self.spatial(x)
        x = x * low_level_feature
        x_0 = x + x1
        x_0 = F.sigmoid(x_0)
        # print("hello pyramid attention block")
        return x_0


class block(nn.Module):
    def __init__(self, input_dims, output_dims, padding=1):
        super(block, self).__init__()
        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size=(3, 3), bias=False, padding=padding)

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=padding)
        self.batchnormalization = BatchNormalization(batch_size=output_dims)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x_mid = x
        x = self.maxpool(x)
        x = self.batchnormalization(x)
        x = self.relu(x)
        return x, x_mid

class convway(nn.Module):
    def __init__(self, output_dim=32, upsample=True, re_co=0.01, channel_num=1, height=28):
        super(convway, self).__init__()
        self.upsample = upsample
        self.re_co = re_co
        self.channel_num = channel_num
        self.padding = (1, 1)
        self.conv1_1 = nn.Conv2d(output_dim, output_dim//2, kernel_size=(1, 1), bias=False, padding=(0, 0))

        self.block_1 = block(output_dim//2, 32)
        self.block_2 = block(32, 64)
        self.block_3 = block(64, 128)
        self.block_4 = block(128, 256)

        self.convc1 = nn.Conv2d(32, output_dim, kernel_size=(1, 1), bias=False, padding=(0, 0))
        self.convc2 = nn.Conv2d(64, output_dim, kernel_size=(1, 1), bias=False, padding=(0, 0))
        self.batch_c1 = BatchNormalization(batch_size=output_dim)
        self.batch_c2 = BatchNormalization(batch_size=output_dim)
        self.upsamp_c1 = UpSamplingBilinear_me(targetsize=height)
        self.upsamp_c2 = UpSamplingBilinear_me(targetsize=height)

        self.convc3 = nn.Conv2d(128, output_dim, kernel_size=(1, 1), bias=False, padding=(0, 0))
        self.convc4 = nn.Conv2d(256, output_dim, kernel_size=(1, 1), bias=False, padding=(0, 0))
        self.batch_c3 = BatchNormalization(batch_size=output_dim)
        self.batch_c4 = BatchNormalization(batch_size=output_dim)
        self.upsamp_c3 = UpSamplingBilinear_me(targetsize=height)
        self.upsamp_c4 = UpSamplingBilinear_me(targetsize=height)
    def forward(self, x):
        output_dim = int(x.shape[1])
        h = int(x.shape[2])
        l2_efficient = 0.01
        kernel_size = [1, 1]
        # Block 1

        filters = output_dim // 2

        x = self.conv1_1(x)
        x, c1 = self.block_1(x)

        # Block 2
        x, c2 = self.block_2(x)

        # Block 3
        x, c3 = self.block_3(x)

        # Block 4
        x, c4 = self.block_4(x)
        if self.upsample:
            c1 = self.convc1(c1)
            c2 = self.convc2(c2)
            c1 = BatchNormalization(batch_size=output_dim)(c1).to(torch.device("cuda"))
            c2 = BatchNormalization(batch_size=output_dim)(c2).to(torch.device("cuda"))
            c1 = UpSamplingBilinear_me(targetsize=h)(c1).to(torch.device("cuda"))
            c2 = UpSamplingBilinear_me(targetsize=h)(c2).to(torch.device("cuda"))
            #print(c1.size())
            #print(c2.size())
            low_level_feature = c1 + c2

            low_level_feature = F.sigmoid(low_level_feature)

            c3 = self.convc3(c3)
            c4 = self.convc4(c4)
            c3 = BatchNormalization(batch_size=output_dim)(c3).to(torch.device("cuda"))
            c4 = BatchNormalization(batch_size=output_dim)(c4).to(torch.device("cuda"))
            c3 = UpSamplingBilinear_me(targetsize=h)(c3).to(torch.device("cuda"))
            c4 = UpSamplingBilinear_me(targetsize=h)(c4).to(torch.device("cuda"))
            high_level_feature = c3 + c4
            high_level_feature = F.sigmoid(high_level_feature)

        return low_level_feature, high_level_feature
