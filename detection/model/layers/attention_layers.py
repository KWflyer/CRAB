import torch
import torch.nn as nn
import torch.nn.functional as F

id = 0
###########################################################################################################
class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_1 = nn.Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0
        )
        self.relu = nn.ReLU(inplace=True)
        self.fc_2 = nn.Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        original = x
        x = self.avg_pool(x)
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.sigmoid(x)
        return original * x


###########################################################################################################
class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_planes = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(
        self, gate_channels, reduction_ratio=16, pool_types=["avg", "max"]
    ):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == "avg":
                avg_pool = F.avg_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == "max":
                max_pool = F.max_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == "lp":
                lp_pool = F.lp_pool2d(
                    x,
                    2,
                    (x.size(2), x.size(3)),
                    stride=(x.size(2), x.size * (3)),
                )
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == "lse":
                # LSE pool
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = (
            torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        )
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)),
            dim=1,
        )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2,
            1,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            relu=False,
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class CBAM(nn.Module):
    def __init__(
        self,
        gate_channels,
        reduction_ratio=16,
        pool_types=["avg", "max"],
        no_spatial=False,
    ):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(
            gate_channels, reduction_ratio, pool_types
        )
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


##############################################################################################################
import math
class eca_attention_od(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(eca_attention_od, self).__init__()
        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(channel, 2) +self.b) / self.gamma)
        k = t if t % 2 else t+1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = torch.sigmoid(y)
        return x * y.expand_as(x)

class eca_attention(nn.Module):
    def __init__(self, k_size=3):
        super(eca_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = torch.sigmoid(y)
        return x * y.expand_as(x)


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
        scale = torch.sigmoid(x_out)  # broadcasting
        #return x * scale
        return scale


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
        x = torch.sigmoid(x)
        batch = x_input.size()[1]
        x = BatchNormalization(batch)(x).to(torch.device("cuda:%d" %id))
        x = UpSamplingBilinear_me(targetsize=height)(x).to(torch.device("cuda:%d" %id))
        x1 = x
        x = self.spatial(x)
        x = x * low_level_feature
        x_0 = x + x1
        # x_0 = torch.sigmoid(x_0)
        # print("hello pyramid attention block")
        return x_0


class block(nn.Module):
    def __init__(self, input_dims, output_dims, padding=1, c=True):
        super(block, self).__init__()
        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size=(3, 3), bias=False, padding=padding)
        self.c = c
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=padding)
        self.batchnormalization = BatchNormalization(batch_size=output_dims)
        self.relu = nn.ReLU()
    def forward(self, x):
        #if self.c == True:
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
        self.conv1_1 = nn.Conv2d(output_dim, output_dim//3, kernel_size=(1, 1), bias=False, padding=(0, 0))

        self.block_1 = block(output_dim//3, 32)
        self.block_2 = block(32, 64)
        self.block_3 = block(64, 128)
        self.block_4 = block(128, 256, c=False)

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
            c1 = BatchNormalization(batch_size=output_dim)(c1).to(torch.device("cuda:%d" %id))
            c2 = BatchNormalization(batch_size=output_dim)(c2).to(torch.device("cuda:%d" %id))
            c1 = UpSamplingBilinear_me(targetsize=h)(c1).to(torch.device("cuda:%d" %id))
            c2 = UpSamplingBilinear_me(targetsize=h)(c2).to(torch.device("cuda:%d" %id))
            #print(c1.size())
            #print(c2.size())
            low_level_feature = c1 + c2

            low_level_feature = torch.sigmoid(low_level_feature)

            c3 = self.convc3(c3)
            c4 = self.convc4(c4)
            c3 = BatchNormalization(batch_size=output_dim)(c3).to(torch.device("cuda:%d" %id))
            c4 = BatchNormalization(batch_size=output_dim)(c4).to(torch.device("cuda:%d" %id))
            c3 = UpSamplingBilinear_me(targetsize=h)(c3).to(torch.device("cuda:%d" %id))
            c4 = UpSamplingBilinear_me(targetsize=h)(c4).to(torch.device("cuda:%d" %id))
            high_level_feature = c3 + c4
            high_level_feature = torch.sigmoid(high_level_feature)

        return low_level_feature, high_level_feature


class UpSamplingBilinear_me(nn.Module):
    def __init__(self, targetsize=32):
        super(UpSamplingBilinear_me, self).__init__()
        self.targetsize = targetsize

    def forward(self, inputs):
        x = nn.UpsamplingBilinear2d(size=self.targetsize)(inputs).to(torch.device("cuda:%d" %id))
        return x