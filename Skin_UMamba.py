from modules.EKAN import *
from modules.CS import *

import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
import math
from mamba_ssm import Mamba

class efem(nn.Module):
    def __init__(self, in_channels, size):
        super(efem, self).__init__()
        self.mamba = ConvSSM(hidden_dim=in_channels)
        self.EKAN_block = EKAN(in_channels, in_channels, size)
        self.layer_norm = nn.LayerNorm([in_channels, size, size])

    def forward(self, feature1):
        feature1 = self.layer_norm(feature1)
        feature1 = feature1.permute(0, 2, 3, 1)
        feature1_1 = self.mamba(feature1)
        feature1_1 = feature1_1.permute(0, 3, 2, 1)
        feature1 = feature1.permute(0, 3, 2, 1)
        feature1_1_1 = F.dropout(feature1_1, p=0.5)
        feature1_1 = self.layer_norm(feature1_1_1)
        feature1_2 = self.EKAN_block(feature1, feature1_1)
        feature1_2 = F.dropout(feature1_2, p=0.5)
        feature1_2 = feature1_2 * feature1_1 + feature1_2
        return feature1_2

class CSIM(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.convlk1 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels*2,7,stride=1,padding=9,dilation=3),
            nn.BatchNorm2d(in_channels*2),
        )
        self.convlk2 = nn.Sequential(
            nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
            nn.BatchNorm2d(1),
        )
        self.maxp = nn.MaxPool2d(2)
        self.avgp = nn.AvgPool2d(2)
    def forward(self,x):
        x_ = self.convlk1(x)
        avgc = torch.mean(x_,dim=1)
        avgc = torch.unsqueeze(avgc,dim=1)
        maxc, _ = torch.max(x_,dim=1)
        maxc = torch.unsqueeze(maxc,dim=1)
        x_ = torch.cat([avgc,maxc],dim=1)
        x_ = self.convlk2(x_)
        avgp_o = self.avgp(x_)
        maxp_o= self.maxp(x_)
        x_ = torch.cat([avgp_o,maxp_o],dim=1)
        x_ = torch.sigmoid(self.convlk2(x_))
        return x_

class BCSIM(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.convlk1 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels*2,7,stride=1,padding=9,dilation=3),
            nn.BatchNorm2d(in_channels*2),
        )
        self.convlk2 = nn.Sequential(
            nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
            nn.BatchNorm2d(1),
        )
        self.maxp = nn.MaxPool2d(2)
        self.avgp = nn.AvgPool2d(2)
    def forward(self,x):
        x_ = self.convlk1(x)
        avgc = torch.mean(x_,dim=1)
        avgc = torch.unsqueeze(avgc,dim=1)
        maxc, _ = torch.max(x_,dim=1)
        maxc = torch.unsqueeze(maxc,dim=1)
        x_ = torch.cat([avgc,maxc],dim=1)
        x_ = self.convlk2(x_)
        x_ = torch.sigmoid(self.convlk2(torch.cat([x_,x_],dim=1)))
        return x_


class Skin_UMamba(nn.Module):

    def __init__(self, num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64],
                 split_att='fc'):
        super().__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )
        self.encoder4 = nn.Sequential(
            efem(in_channels=c_list[2], size=32),
            nn.Conv2d(c_list[2], c_list[3], 1)
        )
        self.encoder5 = nn.Sequential(
            efem(in_channels=c_list[3], size=16),
            nn.Conv2d(c_list[3], c_list[4], 1)
        )
        self.encoder6 = nn.Sequential(
            efem(in_channels=c_list[4], size=8),
            nn.Conv2d(c_list[4], c_list[5], 1)
        )

        self.CSIM3 = CSIM(c_list[2])
        self.CSIM4 = CSIM(c_list[3])
        self.CSIM5 = BCSIM(c_list[4])


        self.decoder1 = nn.Sequential(
            efem(in_channels=c_list[5], size=8),
            nn.Conv2d(c_list[5], c_list[4], 1)
        )
        self.decoder2 = nn.Sequential(
            efem(in_channels=c_list[4], size=8),
            nn.Conv2d(c_list[4], c_list[3], 1)
        )
        self.decoder3 = nn.Sequential(
            efem(in_channels=c_list[3], size=16),
            nn.Conv2d(c_list[3], c_list[2], 1)
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out  # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # b, c1, H/4, W/4

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out  # b, c2, H/8, W/8
        t3_CSIM = self.CSIM3(t3)

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out  # b, c3, H/16, W/16
        t4_CSIM = self.CSIM4(t4)

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out  # b, c4, H/32, W/32
        t5_CSIM = self.CSIM5(t5)

        out = F.gelu(self.encoder6(out))  # b, c5, H/32, W/32
        out = out * t5_CSIM + out

        out5 = F.gelu(self.dbn1(self.decoder1(out)))  # b, c4, H/32, W/32
        out5 = torch.add(out5, t4_CSIM * t5 + t5)

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c3, H/16, W/16
        out4 = torch.add(out4, t3_CSIM * t4 + t4)

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c2, H/8, W/8
        out3 = torch.add(out3, t3)  # b, c2, H/8, W/8

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c1, H/4, W/4
        out2 = torch.add(out2, t2)  # b, c1, H/4, W/4

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c0, H/2, W/2
        out1 = torch.add(out1, t1)  # b, c0, H/2, W/2

        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear',
                             align_corners=True)  # b, num_class, H, W

        return torch.sigmoid(out0)



