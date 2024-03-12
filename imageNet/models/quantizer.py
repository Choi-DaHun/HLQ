import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
import math
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#import matplotlib.pyplot as plt
from torch.autograd import Variable,Function
import math
from torch import linalg as LA

__all__ = ['Conv2dQ']

class Conv2dQ(nn.Conv2d):
    
    def __init__(self, in_channels, out_channels, kernel_size,stride=1, padding=0, dilation=1,groups=1,bias=False, power=True, additive=True, grad_scale=None):
        super(Conv2dQ, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,bias)
        
        self.weight_quant = weight_quantization.apply
        self.act_quant = act_quantization.apply
        self.WL = log_WQ.apply
        self.AL = log_AQ.apply
 
        self.weight_alpha = Parameter(torch.tensor(-0.5))
        self.act_alpha = Parameter(torch.tensor(-0.5))

    def forward(self,x):
        q_w = self.weight_quant(self.weight,self.weight_alpha)
        q_a = self.act_quant(x,self.act_alpha)

        y = F.conv2d(q_a,q_w,self.bias,self.stride ,self.padding,self.dilation,self.groups)
        return y
    def cd(self):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        q_w = self.log_WQ(self.weight).detach()
        out = torch.pow((q_w -self.weight),2).sum()

        return out
    def show_params(self):
        q_w = self.quant(self.weight,self.alpha).detach()
        Q_loss = torch.pow((self.weight -q_w),2).sum()
        print(Q_loss)
        print(self.alpha.data.item())

def Find_Error(Q_E):
    QE_abs = torch.abs(Q_E)
    Emax = torch.max(QE_abs)
    Emin = torch.min(QE_abs)

    Rescale = (QE_abs - Emin)/(Emax - Emin)    
    Error_index = Rescale.gt(0.7).float()
        
    return Error_index

def Grad_scale(Eindex,grad_output,x,q_w,sign):
      

    GEindex = Eindex.lt(1).float()
    grad = Eindex * grad_output
    Eindex = (Eindex * (x - q_w)) 
    grad_norm = torch.norm(grad)
    error_norm = torch.norm(Eindex)
    Eindex = Eindex * grad_norm / error_norm
    grad_scale = torch.abs(grad_output*GEindex + Eindex)*sign  #* g/e  *0.1
    grad_scale = torch.clamp(grad_scale,-0.001,0.001) #defalt 0.01

    return grad_scale
def Log_round(x,alpha):
    xr = x
    r_point = torch.ceil(x)
    r_point = xr - r_point
    point = alpha

    xr[r_point >= point] = torch.ceil(xr[r_point >= point])
    xr[r_point < point] = torch.floor(xr[r_point < point])
    
    xr = torch.clamp(xr,-15,0)
    return xr
class weight_quantization(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,w_alpha):
        sign = torch.sign(x)
              
        x_log = torch.log2(torch.abs(x))
        inf = x_log.gt(-1000).float()
        x_log[inf==0] = -15
        q_w = Log_round(x_log,w_alpha)
        
        q_w = sign * torch.pow(2,q_w) * inf
      
        Error_index = Find_Error(x-q_w)
        ctx.save_for_backward(x,q_w,Error_index,w_alpha)
        return q_w

    @staticmethod
    def backward(ctx,grad_output):
        x,q_w,Eindex,w_alpha = ctx.saved_tensors
        sign = torch.sign(grad_output)
        x_sign= x.sign()
        grad_output = Grad_scale(Eindex,grad_output,x,q_w,sign)
        return grad_output,None

class act_quantization(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,alpha):
        sign = torch.sign(x)
        max_val =  torch.max(x)
        xn = x /max_val
        x_abs = torch.log2(xn) 
        inf = x_abs.gt(-1000).float()
        x_abs[inf==0] = -15
        q_w = Log_round(x_abs,alpha)
        q_w = torch.pow(2,q_w) * sign * max_val  * inf
       
        Quant_Error = x-q_w
             
        Error_index = Find_Error(Quant_Error)
        ctx.save_for_backward(x,q_w,Error_index)
        return q_w

    @staticmethod
    def backward(ctx,grad_output):
        
        x,q_w,Eindex= ctx.saved_tensors
        sign = torch.sign(grad_output)
        x,q_w,Eindex,w_alpha = ctx.saved_tensors
        sign = torch.sign(grad_output)
        x_sign= x.sign()
        grad_output = Grad_scale(Eindex,grad_output,x,q_w,sign)
        return grad_output,None

