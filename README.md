# LSE Kernel Pruning

# This repository contains the codes for our paperID_8872: 
HLQ: Hardware-Friendly Logarithmic Quantization for Power-Efficient Low-Precision CNN Training

## Requirements

   torch (1.8.1)
   torchvision
   numpy
   scipy
   cupy
   cuda

## Dataset Setting
1. CIFAR-10 Model
If you run inference or training through main.py, the dataset will be downloaded automatically.

## Training
```bash
python main.py --epochs 100 --lr 0.01 --data $DATA_PATH$
```

##Unstructured Kernel Pruning
You can find unstruct_kernel_pruning.py in fine_tuning_model/ directory

python unstruct_kernel_pruning.py --path=[weight file path to prune] --save=[directory path to save pruned weight]


## References
 * ImageNet training code : [Pytorch official code](https://github.com/pytorch/examples/blob/main/imagenet/main.py)
 * CIFAR-10 training code : [Pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)
 * CIFAR-10 ResNet reference code : [CIFAR-ResNet](https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py)
