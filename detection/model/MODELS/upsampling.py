import torch
import torch.nn as nn
class UpSamplingBilinear_me(nn.Module):
    def __init__(self, targetsize=32):
        super(UpSamplingBilinear_me, self).__init__()
        self.targetsize = targetsize

    def forward(self, inputs):
        x = nn.UpsamplingBilinear2d(size=self.targetsize)(inputs).to(torch.device("cuda"))
        return x
