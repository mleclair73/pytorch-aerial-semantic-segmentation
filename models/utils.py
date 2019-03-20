from collections import OrderedDict

from torch import nn
import torch.nn.functional as F

class Interpolate(nn.Module):
    def __init__(self, mode, size=None, scale=None, align_corners=True):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.scale = scale
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale, mode=self.mode, align_corners=self.align_corners)
        return x

class BottleNeckPSP(nn.Module):
    def __init__(self, 
                 input_channels, 
                 middle_channels,
                 output_channels,
                 stride,
                 dilation,
                 batch_norm=True,
                 name=None):
        super(BottleNeckPSP, self).__init__()
        bias = not batch_norm
        dilation=max(1, dilation)
        
        self.convolution = nn.Sequential(OrderedDict([
            # 1x1_reduce
            (name + '_1x1_reduce/block', Conv2D_BN_ReLU(name + '_1x1_reduce', 
                                                        input_channels,
                                                        middle_channels,
                                                        kernel_size=1, 
                                                        stride=1,
                                                        padding=0,
                                                        bias=bias,
                                                        include_bn=batch_norm)),
            # 3x3
            (name + '_3x3/block', Conv2D_BN_ReLU(name + '_3x3', 
                                                 middle_channels,
                                                 middle_channels,
                                                 kernel_size=3, 
                                                 stride=stride,
                                                 padding=dilation,
                                                 dilation=dilation,
                                                 bias=bias,
                                                 include_bn=batch_norm)),
            # 1x1_increase
            (name + '_1x1_increase/block', Conv2D_BN(name + '_1x1_increase',
                                                     middle_channels,
                                                     output_channels,
                                                     kernel_size=1,
                                                     stride=1,
                                                     padding=0,
                                                     bias=bias,
                                                     include_bn=batch_norm))
        ]))
        
        self.residual = nn.Sequential(OrderedDict([
            # 1x1_proj
            (name + '_1x1_proj/block',  Conv2D_BN(name + '_1x1_proj',
                                                  input_channels,
                                                  output_channels,
                                                  kernel_size=1,
                                                  stride=stride,
                                                  padding=0,
                                                  bias=bias,
                                                  include_bn=batch_norm))
        ]))
        self.add_module('conv', self.convolution)
        self.add_module('res', self.residual)
    
    def forward(self, inputs):
        return F.relu(self.convolution(inputs) + self.residual(inputs), inplace=True)
    
    
class BottleNeckIdentityPSP(nn.Module):
    def __init__(self, input_channels, middle_channels, stride, dilation, batch_norm=True, name=None):
        super(BottleNeckIdentityPSP, self).__init__()
        bias = not batch_norm
        
        dilation=max(1, dilation)

        self.convolution = nn.Sequential(OrderedDict([
            (name + '_1x1_reduce/block',   Conv2D_BN_ReLU(name + '_1x1_reduce',
                                                          input_channels,
                                                          middle_channels,
                                                          kernel_size=1,
                                                          stride=1,
                                                          padding=0,
                                                          bias=bias,
                                                          include_bn=batch_norm)),
            (name + '_3x3/block',          Conv2D_BN_ReLU(name + '_3x3',
                                                          middle_channels,
                                                          middle_channels,
                                                          kernel_size=3,
                                                          stride=1,
                                                          padding=dilation,
                                                          dilation=dilation,
                                                          bias=bias,
                                                          include_bn=batch_norm)),
            (name + '_1x1_increase/block', Conv2D_BN_ReLU(name + '_1x1_increase',
                                                          middle_channels,
                                                          input_channels,
                                                          kernel_size=1,
                                                          stride=1,
                                                          padding=0,
                                                          bias=bias,
                                                          include_bn=batch_norm)),
        ]))
        self.add_module('conv', self.convolution)

        
    def forward(self, inputs):
        return F.relu(self.convolution(inputs) + inputs, inplace=True)
    
class PyramidPool(nn.Module):
    def __init__(self):
        super(PyramidPool, self).__init__()
        
    def forward(self, inputs):        
        h, w = inputs.shape[2:]
        k_sizes = [(8, 15), (13, 25), (17, 33), (33, 65)]
        strides = [(5, 10), (10, 20), (16, 32), (33, 65)]

        pp_sum = inputs

        for size, stride in zip(k_sizes, strides):
            out = F.avg_pool2d(inputs, size, stride=stride, padding=0)
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
            pp_sum = pp_sum + out
        
        return pp_sum
    
class CascadeFeatureFusion(nn.Module):
    def __init__(self,
                 num_classes,
                 low_res_input_channels,
                 high_res_input_channels,
                 output_channels,
                 batch_norm=True,
                 scale=None,
                 interp=None,
                 high_parent=None):
        super(CascadeFeatureFusion, self).__init__()
        bias = not batch_norm
        
        name = 'conv_sub' + str(scale[0])
        self.low_res = nn.Sequential(OrderedDict([
#             Zoom 2x
            (interp + '_interp', Interpolate(size=2, mode='bilinear', align_corners=True)),
            
            (name + '/block', Conv2D_BN(name,
                                        low_res_input_channels,
                                        output_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=2,
                                        dilation=2,
                                        bias=bias,
                                        include_bn=batch_norm))
        ]))
        # Only used in training
        self.low_res_classifier = nn.Conv2d(low_res_input_channels,
                                            num_classes,
                                            kernel_size=1,
                                            padding=0,
                                            stride=1,
                                            bias=True,
                                            dilation=1)

        name = high_parent + '_sub' + str(scale[1]) + '_proj'
        self.high_res = nn.Sequential(OrderedDict([
            (name + '/block', Conv2D_BN(name,
                                        high_res_input_channels,
                                        output_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=bias,
                                        include_bn=batch_norm))
        ]))
        
    def forward(self, low, high):
        low_scaled = F.interpolate(low, scale_factor=2, mode='bilinear', align_corners=True)
        low_res_classes = self.low_res_classifier(low_scaled)
        low_res_feature_map = self.low_res(low_scaled)

        high_res_feature_map = self.high_res(high)
        h, w = high_res_feature_map.shape[2:]
        low_res_feature_map = F.interpolate(low_res_feature_map, size=(h, w), mode='bilinear', align_corners=True)

        fused_feature_map = F.relu(low_res_feature_map + high_res_feature_map, inplace=True)
        fused_feature_map = F.interpolate(fused_feature_map, scale_factor=2, mode='bilinear', align_corners=True)

        return fused_feature_map, low_res_classes
      
class Conv2D_BN_ReLU(nn.Module):
    def __init__(self, name, input_size, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, include_bn=True):
        super(Conv2D_BN_ReLU, self).__init__()
        
        if include_bn:
            self.block = nn.Sequential(OrderedDict([
                (name, nn.Conv2d(input_size, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, bias=bias)),
                (name + '/bn', nn.BatchNorm2d(out_channels)),
                (name + '/relu', nn.ReLU(inplace=True))
            ]))
        else:
             self.block = nn.Sequential(OrderedDict([
                (name, nn.Conv2d(input_size, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, bias=bias)),
                (name + '/relu', nn.ReLU(inplace=True))
            ]))
        
    def forward(self, inputs):
        return self.block(inputs)
    
class Conv2D_BN(nn.Module):
    def __init__(self, name, input_size, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, include_bn=True):
        super(Conv2D_BN, self).__init__()
        
        if include_bn:
            self.block = nn.Sequential(OrderedDict([
                (name, nn.Conv2d(input_size, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, bias=bias)),
                (name + '/bn', nn.BatchNorm2d(out_channels)),
            ]))
        else:
             self.block = nn.Sequential(OrderedDict([
                (name, nn.Conv2d(input_size, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, bias=bias)),
            ]))
        
    def forward(self, inputs):
        return self.block(inputs)
    
