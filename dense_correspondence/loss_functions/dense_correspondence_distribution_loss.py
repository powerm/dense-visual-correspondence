import torch.nn.functional as F
from numpy import require 
import torch 
from torch.autograd import Variable
import torch.nn as nn

class DenseCorresDistriLoss(object):
    
    def __init__(self, image_shape, config=None):
        self.type = "dense_correspondence_distribution"
        self.image_width = image_shape[1]
        self.image_height = image_shape[0]
        assert config is not None
        self._config = config
        self._debug_data = dict()
        self._debug = False
        self.x = self.get_x()
        self.y = self.get_y()
        
    @property
    def config(self):
        return self._config

    def get_x(self):
        return torch.arange(self.image_width)
    def get_y(self):
        return torch.arange(self.image_height)
    

    def  correspondence_distribution_loss(self, image_a_pred, image_b_pred, matches_a,  matches_b):
        
        
        DCDL= DenseCorresDistriLoss 
        w = self.image_width
        h = self.image_height
        num_matches = matches_a.size()[0]
        x = self.x
        x= x.unsqueeze(0).expand(num_matches, w).unsqueeze(0)
        x= Variable(x.cuda(), requires_grad=False)
        y = self.y
        y= y.unsqueeze(0).expand(num_matches, h).unsqueeze(0)
        y= Variable(y.cuda(), requires_grad=False)

        heatmap_a, heatmap_b=DCDL.heatmap(image_a_pred, image_b_pred, matches_a, matches_b)
        x_a_ex, y_a_ex, x_b_ex, y_b_ex=DCDL.generate_2d_expectation_xy(heatmap_a, heatmap_b, x, y)

        u_a, v_a = matches_a%w, torch.floor(matches_a/w).long()
        u_b, v_b = matches_b%w, torch.floor(matches_b/w).long()

        loss = 1.0/ num_matches *torch.sum(torch.abs(u_a-x_a_ex)+torch.abs(v_a-y_a_ex)+torch.abs(u_b-x_b_ex)+torch.abs(v_b-y_b_ex))
        
        del heatmap_a
        del heatmap_b
        del x_a_ex
        del y_a_ex
        del x_b_ex
        del y_b_ex 
        del u_a
        del v_a
        del u_b
        del v_b
        return loss, loss, zero_loss(), zero_loss(), zero_loss()
    
    #def generate_ex_xy(image_a_pred, image_b_pred, matches_a, matches_b):
        
    
    def heatmap_new(image_a_pred, image_b_pred, matches_a, matches_b):
        
        d= image_a_pred.shape[2]
        num_matches = matches_a.size()[0]
        matches_a_descriptors = torch.index_select(image_a_pred, 1, matches_a)
        matches_b_descriptors = torch.index_select(image_b_pred, 1, matches_b)
        if  len(matches_a) == 1:
            matches_a_descriptors = matches_a_descriptors.unsqueeze(0)
            matches_b_descriptors = matches_b_descriptors.unsqueeze(0)
        matches_a_descriptors_expand = matches_a_descriptors.unsqueeze(2)
        matches_b_descriptors_expand = matches_b_descriptors.unsqueeze(2)
        heatmap_b = None
        heatmap_a = None
        
        for  i  in range(num_matches):
            matches_a_descriptors_i = matches_a_descriptors_expand[:, i]
            norm_diffs =torch.sum(torch.square(image_b_pred- matches_a_descriptors_i), 2).unsqueeze(1)
            heatmap_b_i =  F.softmax(-norm_diffs, 2)
            matches_b_descriptors_i = matches_b_descriptors_expand[:, i]
            norm_diffs =torch.sum(torch.square(image_a_pred- matches_b_descriptors_i), 2).unsqueeze(1)
            heatmap_a_i =  F.softmax(-norm_diffs, 2)
            if i == 0:
                heatmap_b = heatmap_b_i
                heatmap_a = heatmap_a_i
            else:
                heatmap_b = torch.cat((heatmap_b, heatmap_b_i), 1)
                heatmap_a = torch.cat((heatmap_a, heatmap_a_i), 1)
            
            
        assert(heatmap_a.shape[1] == num_matches)
        assert(heatmap_b.shape[1] == num_matches)
            
        return  heatmap_a, heatmap_b
        
        

    @staticmethod
    def heatmap(image_a_pred, image_b_pred, matches_a, matches_b):
        """compute the  response of image_a to  matches_b_descriptors and 
        image_b to matches_a_descriptors  using kernel function -||D(I_a, u_a, I_b, u_b)||_2^2, and softmax to
        [0, 1]
        Args:
            image_a_pred (torch.Variable(torch.FloatTensor)):  image_a_pred: Output of DCN network on image A. has  shape [1, W * H, D]
                image_b_pred (torch.Variable(torch.FloatTensor)): same as image_a_pred
            matches_a (torch.Variable(torch.LongTensor)): has shape [num_matches,],  a (u,v) pair is mapped
            to (u,v) ---> image_width * v + u, this matches the shape of one dimension of image_a_pred
            matches_b (torch.Variable(torch.LongTensor)): same as matches_b
        Returns:
            heatmap_a (torch.FloatTensor) : respense  of  image_a  to  matches_b_descriptors,
                                                        with a torch.Shape([batches, num_matches, h*w])
            heatmap_b (torch.FloatTensor): the same as heatmap_a
            
        """
        
        d= image_a_pred.shape[2]
        num_matches = matches_a.size()[0]
        matches_a_descriptors = torch.index_select(image_a_pred, 1, matches_a)
        matches_b_descriptors = torch.index_select(image_b_pred, 1, matches_b)
        
        if  len(matches_a) == 1:
            matches_a_descriptors = matches_a_descriptors.unsqueeze(0)
            matches_b_descriptors = matches_b_descriptors.unsqueeze(0)
        
        # cal the  image_b respondence distribution for descriptor a 
        
        # expand the image_b_pred from [1, w*h, d] to [1, num_matches, w*h, d]
        image_b_pred_expand =  image_b_pred.unsqueeze(1).expand(1, num_matches, image_b_pred.shape[1], d)
        # expand the matches_a_descriptor from [1,num_maches, d] to [1, num_matches, 1, d]
        matches_a_descriptors_expand = matches_a_descriptors.unsqueeze(2)
        # cal   || D(I_a,u_a, I_b, u_b)||_2^2
        norm_diffs_b = torch.sqrt(torch.sum(torch.pow((image_b_pred_expand- matches_a_descriptors_expand),2),3))
        # cal the  distribution respondence using softmax  exp(-D())/sum_(exp(-D()))
        heatmap_b =  F.softmax(-norm_diffs_b, 2)
        
        image_a_pred_expand =  image_a_pred.unsqueeze(1).expand(1, num_matches, image_a_pred.shape[1], d)
        matches_b_descriptors_expand = matches_b_descriptors.unsqueeze(2)
        norm_diffs_a = torch.sqrt(torch.sum(torch.pow((image_a_pred_expand- matches_b_descriptors_expand),2),3))
        heatmap_a =  F.softmax(-norm_diffs_a, 2)
        
        del norm_diffs_a
        del norm_diffs_b
        
        return  heatmap_a, heatmap_b

    @staticmethod
    def generate_2d_expectation_xy(heatmap_a, heatmap_b, x, y):
        """_summary_

        Args:
            heatmap_a (torch.FloatTensor) : respense  of  image_a  to  matches_b_descriptors,
                                                        with a torch.Shape([batches, num_matches, h*w])
            heatmap_b (torch.FloatTensor): same as heatmap_a
            x (torch.FloatTensor):  x range from [0 ~ w] and with shape [batches, num_matches, w]
            y (torch.FloatTensor): x range from [0 ~ w] and with shape [batches, num_matches, w]

        Returns:
            x_a_expect (torch.FloatTensor) :  expectation of x  from calculate the heatmap distribution  
                                        x expectation, with shape [batches, num_matches]
            y_a_expect (torch.FloatTensor): same as x_a_expect
            x_b_expect (torch.FloatTensor): same as above
            y_b_expect (torch.FloatTensor): same as above
        """
        
        w = x.shape[2]
        h  = y.shape[2]
        
        heatmap_a = heatmap_a.reshape((heatmap_a.shape[0], heatmap_a.shape[1], h,  w))
        heatmap_b = heatmap_b.reshape((heatmap_b.shape[0], heatmap_b.shape[1], h,  w))
        
        heatmap_a_x = torch.sum(heatmap_a, 2)
        heatmap_a_y = torch.sum(heatmap_a, 3)
        
        heatmap_b_x = torch.sum(heatmap_b, 2)
        heatmap_b_y = torch.sum(heatmap_b, 3)
        
        x_a_expect = torch.sum(torch.mul(heatmap_a_x, x),  2)
        y_a_expect = torch.sum(torch.mul(heatmap_a_y, y),  2)
        
        x_b_expect = torch.sum(torch.mul(heatmap_b_x,x),  2)
        y_b_expect = torch.sum(torch.mul(heatmap_b_y,y),  2)
        
        x_a_expect = x_a_expect.squeeze(0)
        y_a_expect = y_a_expect.squeeze(0)
        x_b_expect = x_b_expect.squeeze(0)
        y_b_expect = y_b_expect.squeeze(0)
        
        return x_a_expect, y_a_expect, x_b_expect, y_b_expect

    

def zero_loss():
    return Variable(torch.FloatTensor([0]).cuda())

def is_zero_loss(loss):
    return loss.item() < 1e-20
    
    
    