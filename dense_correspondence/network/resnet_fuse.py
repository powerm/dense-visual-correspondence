import numpy as np
import torch.nn as nn
import torch 
#import torchvision.models as models
import dense_correspondence.network.models as models
#import models


def adjust_input_image_size_for_proper_feature_alignment(input_img_batch, output_stride=8):
    """Resizes the input image to allow proper feature alignment during the
    forward propagation.

    Resizes the input image to a closest multiple of `output_stride` + 1.
    This allows the proper alignment of features.
    To get more details, read here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py#L159

    Parameters
    ----------
    input_img_batch : torch.Tensor
        Tensor containing a single input image of size (1, 3, h, w)

    output_stride : int
        Output stride of the network where the input image batch
        will be fed.

    Returns
    -------
    input_img_batch_new_size : torch.Tensor
        Resized input image batch tensor
    """

    input_spatial_dims = np.asarray( input_img_batch.shape[2:], dtype=np.float )

    # Comments about proper alignment can be found here
    # https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py#L159
    new_spatial_dims = np.ceil(input_spatial_dims / output_stride).astype(np.int) * output_stride + 1

    # Converting the numpy to list, torch.nn.functional.upsample_bilinear accepts
    # size in the list representation.
    new_spatial_dims = list(new_spatial_dims)

    input_img_batch_new_size = nn.functional.upsample_bilinear(input=input_img_batch,
                                                               size=new_spatial_dims)

    return input_img_batch_new_size



class Resnet34_8s_fuse(nn.Module):
    
    
    def __init__(self, num_classes=1000):
        
        super(Resnet34_8s_fuse, self).__init__()
        
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet34_8s_rgb = models.resnet34(fully_conv=True,
                                       pretrained=True,
                                       output_stride=8,
                                       remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet34_8s_rgb.fc = nn.Conv2d(resnet34_8s_rgb.inplanes, num_classes, 1)
        self.resnet34_8s_rgb = resnet34_8s_rgb
        self._normal_initialization(self.resnet34_8s_rgb.fc)
        
        resnet34_8s_depth = models.resnet34(fully_conv=True,
                                       pretrained=True,
                                       output_stride=8,
                                       remove_avg_pool_layer=True)
        resnet34_8s_depth.fc = nn.Conv2d(resnet34_8s_depth.inplanes, num_classes, 1)
        avg =  torch.mean(resnet34_8s_depth.conv1.weight.data, dim=1)
        avg = avg.unsqueeze(1)
        resnet34_8s_depth.conv1 =  nn.Conv2d(1, 64, kernel_size=7,stride=2, padding=3)
        resnet34_8s_depth.conv1.weight.data = avg
        
        self.resnet34_8s_depth = resnet34_8s_depth
        self._normal_initialization(self.resnet34_8s_depth.fc)
    
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
    
    def forward(self, rgb, depth, feature_alignment=False):
        
        input_spatial_dim = rgb.size()[2:]
        
        if feature_alignment:
            
            rgb = adjust_input_image_size_for_proper_feature_alignment(rgb, output_stride=8)
        
        rgb_pre = self.resnet34_8s_rgb(rgb)
        depth_pre = self.resnet34_8s_depth(depth)
        
        rgb_pre = nn.functional.upsample_bilinear(input=rgb_pre, size=input_spatial_dim)
        depth_pre = nn.functional.upsample_bilinear(input=depth_pre, size=input_spatial_dim)
        pre = torch.add(rgb_pre, depth_pre)
        
        return pre



class MS_CAM(nn.Module):
    """
    """ 
    
    def __init__(self, channels =64, r=4):
        super(MS_CAM,self).__init__()
        inter_channels = int(channels // r)
        
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, strider=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1,stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, strider=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl+xg
        wei = self.sigmoid(xlg)
        return x*wei
    
class AFF(nn.Module):
    """multi feature fusion

    Args:
        nn (_type_): _description_
    """
    
        
    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)
        
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        #self.avg =nn.AdaptiveAvgPool2d(1)
        self.global_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        
        self.sigmoid = nn.Sigmoid()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, residual):
        xa = x + residual 
        xl = self.local_att(xa)
        #xg= self.avg(xa)
        xg = self.global_att(xa)
        xlg = xl +xg
        wei = self.sigmoid(xlg)
        
        xo = 2*x*wei + 2*residual*(1-wei)
        
        return xo

class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo



class Resnet34_8s_atten_fuse(nn.Module):
    
    def __init__(self, num_classes=1000):
        
        super(Resnet34_8s_atten_fuse, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet34_8s_rgb = models.resnet34(fully_conv=True,
                                       pretrained=True,
                                       output_stride=8,
                                       remove_avg_pool_layer=True)
        # Randomly initialize the 1x1 Conv scoring layer
        resnet34_8s_rgb.fc = nn.Conv2d(resnet34_8s_rgb.inplanes, resnet34_8s_rgb.inplanes, 1)
        self.resnet34_8s_rgb = resnet34_8s_rgb
        self._normal_initialization(self.resnet34_8s_rgb.fc)
        
        resnet34_8s_depth = models.resnet34(fully_conv=True,
                                       pretrained=True,
                                       output_stride=8,
                                       remove_avg_pool_layer=True)
        resnet34_8s_depth.fc = nn.Conv2d(resnet34_8s_depth.inplanes, resnet34_8s_depth.inplanes, 1)
        avg =  torch.mean(resnet34_8s_depth.conv1.weight.data, dim=1)
        avg = avg.unsqueeze(1)
        resnet34_8s_depth.conv1 =  nn.Conv2d(1, 64, kernel_size=7, stride=2,padding=3)
        resnet34_8s_depth.conv1.weight.data = avg
        self.resnet34_8s_depth = resnet34_8s_depth
        self._normal_initialization(self.resnet34_8s_depth.fc)
        
        self.aff = AFF(channels=resnet34_8s_rgb.inplanes, r=4)
        
        self.last = nn.Conv2d(resnet34_8s_rgb.inplanes, num_classes, 1)
    
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, rgb, depth, feature_alignment=False):
        
        input_spatial_dim = rgb.size()[2:]
        
        if feature_alignment:
            
            rgb = adjust_input_image_size_for_proper_feature_alignment(rgb, output_stride=8)
        
        rgb_pre = self.resnet34_8s_rgb(rgb)
        depth_pre = self.resnet34_8s_depth(depth)
        pre = self.aff(rgb_pre, depth_pre)
        pre = self.last(pre)
        pre = nn.functional.upsample_bilinear(input=pre, size=input_spatial_dim)
        
        return pre
    
    


if __name__ == "__main__":
    
    rgb = torch.rand(1, 3, 480,640)
    depth = torch.rand(1,1, 480, 640)
    net = Resnet34_8s_atten_fuse(num_classes = 3)
    pre = net(rgb, depth)
    print(pre.shape )
    
        
        

        
