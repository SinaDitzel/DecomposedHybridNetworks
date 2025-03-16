''' # This Code was adapted from https://github.com/abdo-eldesokey/nconv
# Licensed under the GPL-3.0 License (https://github.com/abdo-eldesokey/nconv/blob/master/LICENSE)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import collections

from torch.nn.parameter import Parameter
from torch.nn.modules.conv import _ConvNd
#from torch.nn.modules.conv.utils import _pair

from scipy.stats import poisson
from scipy import signal

import numpy as np
import torch.utils.checkpoint as cp
from collections import OrderedDict

from device import device

import math
import time

class NConv2d(_ConvNd):
  
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1), pos_fn='softplus', init_method='p',  padding=(0,0), dilation=(1,1), groups=1, bias=True,padding_mode: str = 'zeros'  ):
        
        # Call _ConvNd constructor
        super(NConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, 0, groups, bias = bias, padding_mode = padding_mode)
        self.use_bias =bias 

        self.eps = 1e-7
        self.pos_fn = pos_fn
        self.init_method = init_method
        
        # Initialize weights and bias
        self.init_parameters()
        
        if self.pos_fn is not None :
            EnforcePos.apply(self, 'weight', pos_fn)
   
        
    def forward(self, data, conf):
        # Normalized Convolution
        denom = F.conv2d(conf, self.weight, None, self.stride,
                        self.padding, self.dilation, self.groups)      
        nomin = F.conv2d(data*conf, self.weight, None, self.stride,
                        self.padding, self.dilation, self.groups)        
        nconv = nomin / (denom+self.eps)
           
        # Add bias
        if self.use_bias:
            b = self.bias
            sz = b.size(0)
            b = b.view(1,sz,1,1)
            b = b.expand_as(nconv)
            nconv += b
        
        # Propagate confidence
        sum_weights = self.weight.sum(dim =[1,2,3]).view(self.weight.shape[0],1,1)
        cout = denom/(sum_weights+self.eps)

        return nconv, cout
    
    
    def init_parameters(self):
        # Init weights
        if self.init_method == 'x': # Xavier            
            torch.nn.init.xavier_uniform_(self.weight)
        elif self.init_method == 'k': # Kaiming
            torch.nn.init.kaiming_uniform_(self.weight)
        elif self.init_method == 'kn':
            torch.nn.init.kaiming_normal_(self.weight)
        elif self.init_method == 'p': # Poisson
            mu=self.kernel_size[0]/2 
            dist = poisson(mu)
            x = np.arange(0, self.kernel_size[0])
            y = np.expand_dims(dist.pmf(x),1)
            w = signal.convolve2d(y, y.transpose(), 'full')
            w = torch.Tensor(w).type_as(self.weight)
            w = torch.unsqueeze(w,0)
            w = torch.unsqueeze(w,1)
            w = w.repeat(self.out_channels, 1, 1, 1)
            w = w.repeat(1, self.in_channels, 1, 1)
            self.weight.data = w + torch.rand(w.shape)
            
        # Init bias
        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(self.out_channels, requires_grad= True)+0.01)
    
    def __repr__(self):
        return "Normalized Convolution(in_channels: %i, out_channels: %i, pos_fn : %s, kernel_size: %i, stride: %i, init: %s, bias: %s"%(self.in_channels, self.out_channels, self.pos_fn, self.kernel_size[0], self.stride[0], self.init_method, self.use_bias)

    def extra_repr(self):
        return ""
    
# Non-negativity enforcement class        
class EnforcePos(object):
    def __init__(self, pos_fn, name):
        self.name = name
        self.pos_fn = pos_fn


    @staticmethod
    def apply(module, name, pos_fn):
        fn = EnforcePos(pos_fn, name)
        
        module.register_forward_pre_hook(fn)                    

        return fn

    def __call__(self, module, inputs):
        weight = getattr(module, self.name)
        weight.data =  F.softplus(weight, beta=10).data
        '''if module.training:
                weight = getattr(module, self.name)
                weight.data = self._pos(weight).data
        else:
                pass
        

    def _pos(self, p):
        pos_fn = self.pos_fn.lower()
        if pos_fn == 'softmax':
            p_sz = p.size()
            p = p.view(p_sz[0],p_sz[1], -1)
            p = F.softmax(p, -1)
            return p.view(p_sz)
        elif pos_fn == 'exp':
            return torch.exp(p)
        elif pos_fn == 'softplus':
            return F.softplus(p, beta=10)
        elif pos_fn == 'sigmoid':
            return F.sigmoid(p)
        else:
            print('Undefined positive function!')
            return'''

class NSequential(nn.Sequential):
    def __init__(self, *args):
        super(NSequential, self).__init__(*args)

    def forward(self,x,c):
        for module in self:
            x,c = module(x,c)
        return x,c

class ConfMaxPool(nn.Module):
    def __init__(self, kernel_size=2):
        super(ConfMaxPool,self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x,c):
        c_ds, idx = F.max_pool2d(c, self.kernel_size[0], self.kernel_size[1], return_indices=True)
        flattened_tensor = x.flatten(start_dim=2)
        x_ds = flattened_tensor.gather(dim=2, index=idx.flatten(start_dim=2)).view_as(idx)
        c_ds /= 4
        return x_ds, c_ds

