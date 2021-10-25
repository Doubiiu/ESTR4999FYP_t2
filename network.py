import torch
import torch.nn.functional as F
from sepConv import SepConv
import torchvision
import random
'''
    This file is the network/model of Video frame interpolation by adaptive sep conv for Course ESTR4999
    Original Paper: Video Frame Interpolation via Adaptive Separable Convolution
    Implementation author: XING Jinbo, 1155091876
'''
class IASC_Kernel_Estimation(torch.nn.Module):
    def __init__(self, kernel_size=51):
        super(IASC_Kernel_Estimation, self).__init__()
        self.kernel_size = kernel_size

        # The Author use convolution aware initialization, here I just use the default initialization method in conv etc. layers

        def CONV_RELU(input_channels, output_channels):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(output_channels, output_channels, 3, 1, 1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(output_channels, output_channels, 3, 1, 1),
                torch.nn.ReLU()
            )

        def AVGPOOL():
            return torch.nn.AvgPool2d(kernel_size=2, stride=2)

        def BILI_UP(channel):
            return torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(channel, channel, 3, 1, 1),
                torch.nn.ReLU()
            )

        def KERNEL(kernel_size):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=kernel_size, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(kernel_size, kernel_size, 3, 1, 1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(kernel_size, kernel_size, 3, 1, 1),
                torch.nn.ReLU(),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(kernel_size, kernel_size, 3, 1, 1)
            )

        # def OCCLUSION():
        #     return torch.nn.Sequential(
        #         torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        #         torch.nn.ReLU(inplace=False),
        #         torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        #         torch.nn.ReLU(inplace=False),
        #         torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        #         torch.nn.ReLU(inplace=False),
        #         torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #         torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
        #         torch.nn.Sigmoid()
        #     )
        self.conv_1 = CONV_RELU(6, 32) # input1&2 RGB*2=6channels
        self.avgpool_1 = AVGPOOL()

        self.conv_2 = CONV_RELU(32, 64)
        self.avgpool_2 = AVGPOOL()

        self.conv_3 = CONV_RELU(64, 128)
        self.avgpool_3 = AVGPOOL()

        self.conv_4 = CONV_RELU(128, 256)
        self.avgpool_4 = AVGPOOL()

        self.conv_5 = CONV_RELU(256, 512)
        self.avgpool_5 = AVGPOOL()

        self.conv_6 = CONV_RELU(512, 512)

        self.biupsample_1 = BILI_UP(512)
        self.conv_7 = CONV_RELU(512, 256)

        self.biupsample_2 = BILI_UP(256)
        self.conv_8 = CONV_RELU(256, 128)

        self.biupsample_3 = BILI_UP(128)
        self.conv_9 = CONV_RELU(128, 64)

        self.biupsample_4 = BILI_UP(64)

        self.k1Vertical = KERNEL(self.kernel_size)
        self.k1Horizontal = KERNEL(self.kernel_size)
        self.k2Vertical = KERNEL(self.kernel_size)
        self.k2Horizontal = KERNEL(self.kernel_size)
        # self.occlusion = OCCLUSION()

    def forward(self, input):
        Conv_1 = self.conv_1(input)
        AvgPool_1 = self.avgpool_1(Conv_1)

        Conv_2 = self.conv_2(AvgPool_1)
        AvgPool_2 = self.avgpool_2(Conv_2)

        Conv_3 = self.conv_3(AvgPool_2)
        AvgPool_3 = self.avgpool_3(Conv_3)

        Conv_4 = self.conv_4(AvgPool_3)
        AvgPool_4 = self.avgpool_4(Conv_4)

        Conv_5 = self.conv_5(AvgPool_4)
        AvgPool_5 = self.avgpool_5(Conv_5)

        Conv_6 = self.conv_6(AvgPool_5)

        Biupsample_1 = self.biupsample_1(Conv_6)
        Skip1 = Biupsample_1 + Conv_5 # skip connection
        Conv_7 = self.conv_7(Skip1)

        Biupsample_2 = self.biupsample_2(Conv_7)
        Skip2 = Biupsample_2 + Conv_4
        Conv_8 = self.conv_8(Skip2)

        Biupsample_3 = self.biupsample_3(Conv_8)
        Skip3 = Biupsample_3 + Conv_3
        Conv_9 = self.conv_9(Skip3)

        Biupsample_4 = self.biupsample_4(Conv_9)
        Skip4 = Biupsample_4 + Conv_2

        K1Vertical = self.k1Vertical(Skip4)
        K1Horizontal = self.k1Horizontal(Skip4)
        K2Vertical = self.k2Vertical(Skip4)
        K2Horizontal = self.k2Horizontal(Skip4)
        # Occ = self.occlusion(Skip4)
        # return K1Vertical, K1Horizontal, K2Vertical, K2Horizontal, Occ
        return K1Vertical, K1Horizontal, K2Vertical, K2Horizontal

class IASC(torch.nn.Module):
    def __init__(self, kernel_size=51, lr=0.001):
        super(IASC, self).__init__()
        self.kernel_size = kernel_size
        self.kernel = IASC_Kernel_Estimation(self.kernel_size)
        self.pad = torch.nn.ReplicationPad2d(self.kernel_size//2)

    def forward(self, frame_f, frame_l): #frame_f==first frame, frame_l===third frame, aim to get frame_i, that is second frame

        assert(frame_f.size(2) == frame_f.size(2))
        assert(frame_f.size(3) == frame_f.size(3))
        combine_frame = torch.cat([frame_f, frame_l], dim=1) # cat two frames in channel dimension
        # kv1, kh1, kv2, kh2, occ = self.kernel(combine_frame)
        kv1, kh1, kv2, kh2 = self.kernel(combine_frame)
        # local separate convolution for two input frames
        # frame_i = occ * SepConv().apply(self.pad(frame_f), kv1, kh1) + (1 - occ) * SepConv().apply(self.pad(frame_l), kv2, kh2)
        frame_i = SepConv().apply(self.pad(frame_f), kv1, kh1) + SepConv().apply(self.pad(frame_l), kv2, kh2)
        # torchvision.utils.save_image(occ, 'occlusion_map'+str(random.random())+'.png', range=(0, 1))
        return frame_i

class Overlap(torch.nn.Module):
    def __init__(self):
        super(Overlap, self).__init__()

    def forward(self, frame_f, frame_l):
        return (frame_f*0.5 + frame_l*0.5)
