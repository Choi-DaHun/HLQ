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

## Inference
python main -e --resume=[weight file path to inference]

##Unstructured Kernel Pruning
You can find unstruct_kernel_pruning.py in fine_tuning_model/ directory

python unstruct_kernel_pruning.py --path=[weight file path to prune] --save=[directory path to save pruned weight]


## References
 * ImageNet training code : [https://github.com/pytorch/examples/blob/main/imagenet/main.py]
