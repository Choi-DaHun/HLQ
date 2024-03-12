# LSE Kernel Pruning

# This repository contains the codes for our paperID_8872: 
HLQ: Hardware-Friendly Logarithmic Quantization for Power-Efficient Low-Precision CNN Training
### HLQ

## Requirements

   torch (1.8.1)
   torchvision
   numpy
   scipy
   cupy
   cuda

## Dataset Setting
1. CIFAR-10/100
If you run inference or training through main.py, the dataset will be downloaded automatically.
2. ImageNet
You can install imagenet dataset file [ImageNet dataset](https://www.image-net.org/download)

## Training
`models.quantizer.py' contains the configuration for quantization. In particular, you can specify them in the class `Conv2dQ`:
```python
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
```
You can change the initial rounding point by modifying the alpha parameter.


```bash
python main.py --epochs 100 --lr 0.01 --data $DATA_PATH$
```


## References
 * ImageNet training code : [Pytorch official code](https://github.com/pytorch/examples/blob/main/imagenet/main.py)
 * CIFAR-10 training code : [Pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)
 * CIFAR-10 ResNet reference code : [CIFAR-ResNet](https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py)
