# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Script for u-net architecture. 

@author: Uroš Perkan

Changes:

"""

# NOTE: model works only for even number dimensions
# e.g. 180x360 or 60x120.

# LIBRARIES

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np

# CONVOLUTION LAYERS
# --> Add 5x5 convolution for first layer
# --> padding = "same"
# --> padding_mode = "zeros", "reflect", "replicate", "circular"
# --> bias = True
# --> groups = 1 

# --> BatchNormalization could be usefull!

# --> nn.ELU or nn.GLU
# --> LSTM cell in internal state

# NOTE ON CONVOLUTION
# https://www.youtube.com/watch?v=KTB_OFoAQcc
# One filter shape is kernel_size x kernel_size x in_channels
# and it acts on all input channels. Each channel is multiplied
# with the filter of it's channel and then the sum of all the convolved
# numbers is one number in result. Only if you have N filters you
# will then get N outputs and therefore out_channel = N. 

# You can think of a filter of a box size kernel_size x kernel_size x in_channels,
# which is put on the channels tensor, makes componentwise multiplication and
# sums it all up in one number on i,j position of 1 new channel.

padding_9 = 3
def conv7x7(in_channels, out_channels, stride=1, 
            padding=0, bias=True, groups=1):    
    """3x3 convolution layer"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=7,
        stride=stride,
        padding=padding,
        bias=bias,
        padding_mode='zeros',
        groups=groups)

padding_3 = 1
def conv3x3(in_channels, out_channels, stride=1, 
            padding=0, bias=True, groups=1):    
    """3x3 convolution layer"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        padding_mode='zeros',
        groups=groups)

# padding_1 = 0
# def conv1x1(in_channels, out_channels, stride=1, 
#             padding=0, bias=True, groups=1):    
#     """1x1 convolution layer"""
#     return nn.Conv2d(
#         in_channels,
#         out_channels,
#         kernel_size=1,
#         stride=stride,
#         padding=padding,
#         bias=bias,
#         padding_mode='zeros',
#         groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose', output_padding=(0,0)):
    """2x2 upsampling"""
    # Convolutional upsampling
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            output_padding=output_padding)
    else:
        # out_channels is always going to be the same
        # as in_channels
        # Linear interpolation in upsampling
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))

def conv1x1(in_channels, out_channels, groups=1):
    # Change dimentionality of filtered space
    # (N,F_in,H,W) -> (N,F_out,H,W)
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


def Pad(fourdtensor, EastWest_pad, NorthSouth_pad):
    
    # NORTH AND SOUTH PADDING
    if NorthSouth_pad > 0:
        # Top of matrix transformation
        top = torch.flip(fourdtensor[:,:,0:NorthSouth_pad,:], dims=(-2,))
        top = torch.roll(top, shifts=int(top.shape[-1]/2), dims=-1)
        # Bottom of matrix transformation
        bottom = torch.flip(fourdtensor[:,:,-NorthSouth_pad:,:], dims=(-2,))
        bottom = torch.roll(bottom, shifts=int(bottom.shape[-1]/2), dims=-1)
        # Stack together
        arr = torch.concat((top, fourdtensor, bottom), dim=-2)
    else:
        arr = fourdtensor

    # EAST AND WEST PADDING
    if EastWest_pad == 0:
        return arr
    else:
        left = arr[:,:,:,0:EastWest_pad]
        right = arr[:,:,:,-EastWest_pad:]
        arr = torch.concat((right, arr, left),dim=-1)
        return arr

class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, depth, pooling=True):
        super(DownConv, self).__init__() # DownConv inherits Module methods 

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        if depth <= 3:
            self.conv1 = conv7x7(self.in_channels, self.out_channels)
            self.conv2 = conv7x7(self.in_channels + self.out_channels, self.out_channels)
            self.conv3 = conv7x7(self.in_channels + self.out_channels, self.out_channels)
            self.conv4 = conv7x7(self.in_channels + self.out_channels, self.out_channels)
        else:
            self.conv1 = conv3x3(self.in_channels, self.out_channels)
            self.conv2 = conv3x3(self.in_channels + self.out_channels, self.out_channels)
            self.conv3 = conv3x3(self.in_channels + self.out_channels, self.out_channels)
            self.conv4 = conv3x3(self.in_channels + self.out_channels, self.out_channels)

        # elif depth < 6:
        #     self.conv1 = conv3x3(self.in_channels, self.out_channels)
        #     self.conv2 = conv3x3(self.in_channels + self.out_channels, self.out_channels)
        #     self.conv3 = conv3x3(self.in_channels + self.out_channels, self.out_channels)
        #     self.conv4 = conv3x3(self.in_channels + self.out_channels, self.out_channels)
        # else:
        #     self.conv1 = conv1x1(self.in_channels, self.out_channels)
        #     self.conv2 = conv1x1(self.in_channels + self.out_channels, self.out_channels)
        #     self.conv3 = conv1x1(self.in_channels + self.out_channels, self.out_channels)
        #     self.conv4 = conv1x1(self.in_channels + self.out_channels, self.out_channels)

        self.BatchNorm1 = nn.BatchNorm2d(self.out_channels)
        self.BatchNorm2 = nn.BatchNorm2d(self.out_channels)
        self.BatchNorm3 = nn.BatchNorm2d(self.out_channels)
        self.BatchNorm4 = nn.BatchNorm2d(self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.LeakyReLU = nn.LeakyReLU()

    def forward(self, x, depth, p):
        
        if depth <= 3:
            padding = padding_9
        else:
            padding = padding_3
        # elif depth < 6:
        #     padding = padding_3
        # else:
        #     padding = padding_1

        x0 = x
        x = Pad(x, padding, padding)
        x = self.BatchNorm1(self.LeakyReLU(self.conv1(x)))
        x = F.dropout(x, p)

        x = torch.cat((x0, x), 1)
        x = Pad(x, padding, padding)
        x = self.BatchNorm2(self.LeakyReLU(self.conv2(x)))
        x = F.dropout(x, p)
        
        x = torch.cat((x0, x), 1)
        x = Pad(x, padding, padding)
        x = self.BatchNorm3(self.LeakyReLU(self.conv3(x)))
        x = F.dropout(x, p)

        x = torch.cat((x0, x), 1)
        x = Pad(x, padding, padding)
        x = self.BatchNorm4(self.LeakyReLU(self.conv4(x)))
        x = F.dropout(x, p)

        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, depth,
                 merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__() # UpConv inherits Module methods

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        OUTPUT_PADDING = [
            (1,1), # depth=0 (5)
            (1,0), # depth=1 (4)
            (0,1), # depth=2 (3)
            (1,0), # depth=3 (2)
            (0,0), # depth=4 (1)
            (0,0), # depth=5 (0)
        ]

        self.upconv = upconv2x2(
            self.in_channels,
            self.out_channels, 
            mode=self.up_mode,
            output_padding=OUTPUT_PADDING[depth])

        # self.upconv = upconv2x2(self.in_channels, self.out_channels, 
        #     mode=self.up_mode)

        if self.merge_mode == 'concat':
            # Takes 2 concatenated tensors and performs convolution
            # which by default acts on all 2*out_channels therefore
            # each resulting layer is convolution of encoding and decoding
            # tensor so some input information is restored.
            self.conv1 = conv7x7(2*self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv7x7(self.out_channels, self.out_channels)

        self.conv2 = conv7x7(2*self.out_channels, self.out_channels)
        self.conv3 = conv7x7(2*self.out_channels, self.out_channels)
        self.conv4 = conv7x7(2*self.out_channels, self.out_channels)

        self.BatchNorm1 = nn.BatchNorm2d(self.out_channels)
        self.BatchNorm2 = nn.BatchNorm2d(self.out_channels)
        self.BatchNorm3 = nn.BatchNorm2d(self.out_channels)
        self.BatchNorm4 = nn.BatchNorm2d(self.out_channels)

        self.LeakyReLU = nn.LeakyReLU()


    def forward(self, from_down, from_up, depth, p):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        # First upsample tensor from decoder pathway
        from_up = self.upconv(from_up)
        # Merge upsampled tensor from decoder with encoder
        if self.merge_mode == 'concat':
            # Concatenate encoding and decoding tensor
            x = torch.cat((from_up, from_down), 1)
        else:
            # Elementwise summation of encoding and decoding tensor
            x = from_up + from_down 
        
        # if depth < 2:
        #     padding = padding_3
        # else:
        padding = padding_9

        x0 = from_up
        x = Pad(x, padding, padding)
        x = self.BatchNorm1(self.LeakyReLU(self.conv1(x)))
        x = F.dropout(x, p)

        x = torch.cat((x0, x), 1)        
        x = Pad(x, padding, padding)
        x = self.BatchNorm2(self.LeakyReLU(self.conv2(x)))
        x = F.dropout(x, p)

        x = torch.cat((x0, x), 1)
        x = Pad(x, padding, padding)
        x = self.BatchNorm3(self.LeakyReLU(self.conv3(x)))
        x = F.dropout(x, p)

        x = torch.cat((x0, x), 1)
        x = Pad(x, padding, padding)
        x = self.BatchNorm4(self.LeakyReLU(self.conv4(x)))
        x = F.dropout(x, p)
                
        return x

class UNet(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597
    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).
    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, in_channels, out_channels, depth=5, 
                 start_filts=64, up_mode='transpose', 
                 merge_mode='concat'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the 
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))
    
        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts#*(2**i)
            pooling = True if i < depth-1 else False

            # Get x, before_pool from encoding block
            down_conv = DownConv(ins, outs, i, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth-1):
            ins = outs
            outs = ins# // 2
            
            # Get x from decoding block
            up_conv = UpConv(ins, outs, i, up_mode=up_mode,
                merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        # Final convolution - number of channels must match number
        # of initial channels since they should be the same fields
        # !!!!!!!!!! Is here really convolution?!!!!!!!!!!!!!!!!!!
        # self.conv_final = conv1x1(outs, self.in_channels)
        self.conv_final = conv1x1(outs, self.out_channels)


        # add the list of modules to current module
        # ModuleList holds submodels in a list
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

    def __repr__(self):
        """
        Method for recreating the object. You can acces it by
        typing 'print(object)' or 'repr(object)'.
        """
        return f"UNet(in_ch-'{self.in_channels}', out_ch-'{self.out_channels}', depth'{self.depth}', start filters'{self.start_filts}', up mode'{self.up_mode}', merge mode'{self.merge_mode}')"

    # Create function that initializes weights as normal distributed
    # values that are not to big and not to small.
    @staticmethod # Doesn't take neither self nor cls as input
    def weight_init(m): # m stands for module
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight) # Normaly distribute weights of convolution layers
            init.constant_(m.bias, 0) # Set all biases to zero

    # Method for initializing weights in all modules
    def reset_params(self):
        # NOTE self.modules is an iterable through "nn.ModuleList"
        # or more precisely through self.down_convs and self.up_convs
        for i, m in enumerate(self.modules()): 
            self.weight_init(m)

    def forward(self, x, p=0.):
        # List of tensors from encoding part, which will be used for
        # skipped connections.
        encoder_outs = []
         
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x, i, p)
            #print("Encoder outs", x.shape)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x, i, p)
            #print("Decoder outs", x.shape)

        # Here you can add an activation function if you want        
        x = self.conv_final(x)
        return x

if __name__ == "__main__":
    """
    Testing if it returns a result
    """
    import resource
    
    model = UNet(in_channels=1, depth=5, start_filts=30, up_mode='transpose', merge_mode='concat')
    x = Variable(torch.FloatTensor(np.random.random((1, 1, 60, 120))))
    out = model(x)
    loss = torch.sum(out)
    loss.backward()

    print(loss)
    print(out.shape)
    print(out.shape)
    print(model)
    print(repr(model))
    print(f"Training max {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024} MB")
