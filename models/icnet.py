from collections import OrderedDict

from torch import nn
import torch.nn.functional as F

from models.utils import *

# Now we can define our network
class ICNet(nn.Module):
    """
    https://arxiv.org/pdf/1704.08545.pdf
    """
    def __init__(self, num_classes, batch_norm=True):
        super(ICNet, self).__init__()
        self.num_classes = num_classes
        bias = not batch_norm
                
        self.full_scale = nn.Sequential(
            OrderedDict([
            ('conv1_sub1/block', 
             Conv2D_BN_ReLU('conv1_sub1',  3, 32, kernel_size=3, padding=1, stride=2, bias=bias, include_bn=batch_norm)),
            ('conv2_sub1/block',
             Conv2D_BN_ReLU('conv2_sub1', 32, 32, kernel_size=3, padding=1, stride=2, bias=bias, include_bn=batch_norm)),
            ('conv3_sub1/block',
             Conv2D_BN_ReLU('conv3_sub1', 32, 64, kernel_size=3, padding=1, stride=2, bias=bias, include_bn=batch_norm))
        ]))
        self.half_scale = nn.Sequential(OrderedDict([
            ('data_sub2', Interpolate(scale=0.5, mode='bilinear', align_corners=True)),
            
            ('conv1_1_3x3_s2/block',      Conv2D_BN_ReLU('conv1_1_3x3_s2', 3, 32, kernel_size=3, padding=1, stride=2, bias=bias, include_bn=batch_norm)),
            ('conv1_2_3x3/block',         Conv2D_BN_ReLU('conv1_2_3x3',   32, 32, kernel_size=3, padding=1, stride=1, bias=bias, include_bn=batch_norm)),
            ('conv1_3_3x3/block',         Conv2D_BN_ReLU('conv1_3_3x3',   32, 64, kernel_size=3, padding=1, stride=1, bias=bias, include_bn=batch_norm)),

            ('pool1_3x3_s2',        nn.MaxPool2d(3, stride=2, padding=1)),

            # residual block 1
            ('res1_1',     BottleNeckPSP(64, 32, 128, stride=1, dilation=1, batch_norm=batch_norm, name='conv2_1')),
            ('res1_2', BottleNeckIdentityPSP(128, 32, stride=1, dilation=1, batch_norm=batch_norm, name='conv2_2')),
            ('res1_3', BottleNeckIdentityPSP(128, 32, stride=1, dilation=1, batch_norm=batch_norm, name='conv2_3')),
            
            # residual block 2 (first half bottleneck)
            ('res2_1',     BottleNeckPSP(128, 64, 256, stride=2, dilation=1, batch_norm=batch_norm, name='conv3_1'))
        ]))
        self.quarter_scale = nn.Sequential(OrderedDict([
            ('conv3_1_sub4', Interpolate(scale=0.5, mode='bilinear', align_corners=True)),
            
            # Starts with second half of residual block 2
            ('res_2_2',  BottleNeckIdentityPSP(256, 64, stride=1, dilation=1, batch_norm=batch_norm, name='conv3_2')),
            ('res_2_3',  BottleNeckIdentityPSP(256, 64, stride=1, dilation=1, batch_norm=batch_norm, name='conv3_3')),
            ('res_2_4',  BottleNeckIdentityPSP(256, 64, stride=1, dilation=1, batch_norm=batch_norm, name='conv3_4')),
            
            # Residual Block 3
            ('res_3_1',           BottleNeckPSP(256, 128, 512, 1, dilation=2, batch_norm=batch_norm, name='conv4_1')),
            
            ('res_3_2', BottleNeckIdentityPSP(512, 128, stride=1, dilation=2, batch_norm=batch_norm, name='conv4_2')),
            ('res_3_3', BottleNeckIdentityPSP(512, 128, stride=1, dilation=2, batch_norm=batch_norm, name='conv4_3')),
            ('res_3_4', BottleNeckIdentityPSP(512, 128, stride=1, dilation=2, batch_norm=batch_norm, name='conv4_4')),
            ('res_3_5', BottleNeckIdentityPSP(512, 128, stride=1, dilation=2, batch_norm=batch_norm, name='conv4_5')),
            ('res_3_6', BottleNeckIdentityPSP(512, 128, stride=1, dilation=2, batch_norm=batch_norm, name='conv4_6')),

            #Residual Block 4
            ('res_4_1',    BottleNeckPSP(512, 256, 1024, 1, dilation=4, batch_norm=batch_norm, name='conv5_1')),
            ('res_4_2', BottleNeckIdentityPSP(1024, 256, 1, dilation=4, batch_norm=batch_norm, name='conv5_2')),
            ('res_4_3', BottleNeckIdentityPSP(1024, 256, 1, dilation=4, batch_norm=batch_norm, name='conv5_3')),
            ('pyramid_pool', PyramidPool()),
        
            #Conv 5_4_k1
            ('conv5_4_k1',      Conv2D_BN_ReLU('conv5_4_k1', 1024, 256, kernel_size=1, padding=0, stride=1, bias=bias, include_bn=batch_norm)),
        ]))
        self.quarter_half_CFF = CascadeFeatureFusion(num_classes,
                                                     256,
                                                     256,
                                                     128,
                                                     scale=(4,2),
                                                     interp='conv5_4',
                                                     high_parent='conv3_1',
                                                     batch_norm=batch_norm)
        self.half_full_CFF = CascadeFeatureFusion(num_classes,
                                                  128,
                                                  64,
                                                  128,
                                                  scale=(2,1),
                                                  interp='sub24_sum',
                                                  high_parent='conv3',
                                                  batch_norm=batch_norm)
        self.conv6_cls = nn.Conv2d(128, num_classes, 1, stride=1, padding=0)
    def forward(self, inputs):
        h, w = inputs.shape[:2]

        full_features = self.full_scale(inputs)
        half_features = self.half_scale(inputs)   
        quarter_features = self.quarter_scale(half_features)
        
        quarter_half_fused, quarter_classes = self.quarter_half_CFF(quarter_features, half_features)
#         quarter_half_fused = F.interpolate(quarter_half_fused, scale_factor=2, mode='bilinear', align_corners=True)
        all_scales_fused, half_classes = self.half_full_CFF(quarter_half_fused, full_features)
        all_scales_fused = F.interpolate(all_scales_fused, scale_factor=2, mode='bilinear', align_corners=True)
        # print(all_scales_fused.shape)
        all_scale_classes = self.conv6_cls(all_scales_fused)
        # print(all_scale_classes.shape)
        
        if self.training:
            return (all_scale_classes, half_classes, quarter_classes)  
        else:
            return F.interpolate(all_scale_classes, scale_factor=4, mode='bilinear', align_corners=True)
