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
        resnet34_8s_depth.conv1 =  nn.Conv2d(1, 64, kernel_size=3, padding=1)
        resnet34_8s_depth.conv1.weight.data = avg
        
        self.resnet34_8s_depth = resnet34_8s_depth
        self._normal_initialization(self.resnet34_8s_depth.fc)
    
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
    
    def forward(self, rgb, depth, feature_alignment=False):
        
        input_spatial_dim = rgb.size()[2:]
        
        if feature_alignment:
            
            rgb = adjust_input_image_size_for_proper_feature_alignment(x, output_stride=8)
        
        rgb_pre = self.resnet34_8s_rgb(rgb)
        depth_pre = self.resnet34_8s_depth(depth)
        
        rgb_pre = nn.functional.upsample_bilinear(input=rgb_pre, size=input_spatial_dim)
        depth_pre = nn.functional.upsample_bilinear(input=depth_pre, size=input_spatial_dim)
        pre = torch.add(rgb_pre, depth_pre)
        
        return pre
    

if __name__ == "__main__":
    
    rgb = torch.rand(1,3,480,640)
    
    depth = torch.rand(1,1,480,640)
    net = Resnet34_8s_fuse(num_classes = 3)
    pre = net(rgb, depth)
    print(pre)
    
        
        

        
