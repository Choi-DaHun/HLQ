# LSE Kernel Pruning

# This repository contains the codes for our paperID_7312: 
Layer Sensitivity Estimation-based Adaptive Kernel Pruning for Accelerating Deep Convolutional Neural Networks

## Requirements

   torch (1.8.1)
   torchvision
   numpy
   scipy
   cupy
   cuda

## Dataset Setting
1. CIFAR-10 Model
1-1. ResNet
If you run inference or training through main.py, the dataset will be downloaded automatically.

1.2 MobileNetV2
If you run inference or training through main.py, the dataset will be downloaded automatically.
 

2. ImageNet
2-1. ResNet
1) Trainset
Modify the path of traindir = () in data_loader.py to match the imagenet trainset

2) Validset
Modify the path of valdir = () in data_loader.py to match the imagenet validset


2-2. MobileNetV2
1) Trainset
Modify the path of traindir = () in data_loader.py to match the imagenet trainset

2) Validset
Modify the path of valdir = () in data_loader.py to match the imagenet validset

##Weights Setting
CIFAR-10 MobileNetV2  | Fine-tuning Model | Baseline weight is in weights/ directory
CIFAR-10 MobileNetV2  | Fine-tuning Model | Fine-tuned weight is in fine_tuned/ directory
CIFAR-10 MobileNetV2  | Typical Conv Model | This Model can share weights with Fine-tuning Model
CIFAR-10 MobileNetV2  | Channel Index Conv | Structured kernel pruned weight is in structured/ directory
CIFAR-10 MobileNetV2  | Channel Index Conv | Encoded weight is in encoded/ directory
CIFAR-10 MobileNetV2  | Channel Index Conv | Decoded weight is in decoded/ directory
-----------------------------------------------------------------------------------------------------------------------
CIFAR-10 ResNet-56 | Fine_tuning_Model | Baseline weight is in weights/ directory
CIFAR-10 ResNet-56 | Fine_tuning_Model | Fine-tuned weight is in fine_tuned/resnet56/ directory
CIFAR-10 ResNet-56  | Typical Conv Model | This Model can share weights with Fine-tuning Model
CIFAR-10 ResNet-56  | Channel Index Conv | Structured kernel pruned weight is in structured/ directory
CIFAR-10 ResNet-56  | Channel Index Conv | Encoded weight is in encoded/resnet56/ directory
CIFAR-10 ResNet-56  | Channel Index Conv | Decoded weight is in decoded/resnet56/ directory
-----------------------------------------------------------------------------------------------------------------------
CIFAR-10 ResNet-110 | Fine_tuning_Model | Baseline weight is in weights/ directory
CIFAR-10 ResNet-110 | Fine_tuning_Model | Fine-tuned weight is in fine_tuned/resnet110/ directory
CIFAR-10 ResNet-110 | Typical Conv Model | This Model can share weights with Fine-tuning Model
CIFAR-10 ResNet-110 | Channel Index Conv | Structured kernel pruned weight is in structured/ directory
CIFAR-10 ResNet-110 | Channel Index Conv | Encoded weight is in encoded/resnet110/ directory
CIFAR-10 ResNet-100 | Channel Index Conv | Decoded weight is in decoded/resnet110/ directory
-----------------------------------------------------------------------------------------------------------------------
ImageNet ResNet-50 | Fine-tuning Model | Baseline weight can be downloaded in : https://drive.google.com/file/d/1A5ytLSwYcOZumNhIxuLDP4ozVWQ6lV8y/view?usp=sharing
ImageNet ResNet-50 | Fine-tuning Model | Fine-tuned weight can be downloaded in : https://drive.google.com/file/d/1nFEbeBfOu2Qz0LZlIVT0YpFkBggY9NRI/view?usp=sharing
ImageNet ResNet-50 | Typical Conv Model | This Model can share weights with Fine-tuning Model
ImageNet ResNet-50 | Channel Index Conv | Structured kernel pruned weight can be downloaded in : https://drive.google.com/file/d/1_E-t4jfIBqU5XdWCycuqhMrutqp5guec/view?usp=sharing
ImageNet ResNet-50 | Channel Index Conv | Encoded weight can be downloaded in : https://drive.google.com/file/d/1vhVG9Te1E9oSQBChj-JXcciPAzFbSTQ1/view?usp=sharing
ImageNet ResNet-50 | Channel Index Conv | Decoded weight can be downloaded in : https://drive.google.com/file/d/1CA9My8x14CwrEYXs8ZSCzq2VxxiIG9yJ/view?usp=sharing
-----------------------------------------------------------------------------------------------------------------------
ImageNet MobileNetV2 | Fine-tuning Model | Baseline weight can be downloaded in : https://drive.google.com/file/d/1WRKszjE3XVo2uEMbL38U1d_R6HZkKr5K/view?usp=sharing
ImageNet MobileNetV2 | Fine-tuning Model | Fine-tuned weight can be downloaded in : https://drive.google.com/file/d/15LDHDJLjeZJ0zi3aApBNWvotTzOC3OEw/view?usp=sharing
ImageNet MobileNetV2 | Typical Conv Model | This Model can share weights with Fine-tuning Model
ImageNet MobileNetV2 | Channel Index Conv | Structured kernel pruned weight can be downloaded in : https://drive.google.com/file/d/1Nezo67VR-Sm_40mCOp06P0gBiTH3flw6/view?usp=sharing
ImageNet MobileNetV2 | Channel Index Conv | Encoded weight can be downloaded in : https://drive.google.com/file/d/1X9vd1910EFKmsIDhtogU0Z4MMTCq1xoX/view?usp=sharing
ImageNet MobileNetV2 | Channel Index Conv | Decoded weight can be downloaded in : https://drive.google.com/file/d/1LkfBqhYJt7Vmkz4CUFTvJg8B2nOwK3dl/view?usp=sharing
-----------------------------------------------------------------------------------------------------------------------

##Directory
fine_tuning_model : A normal Pytorch Model for fine-tuning unstructured kernel pruned weight
typical_conv_model : A typical convolution model implemented in cuda code for comparison
channel_index_model : A model to which Channel Index Convolution implemented in cuda code is applied


##Inference

python main -e --resume=[weight file path to inference]

##Unstructured Kernel Pruning
You can find unstruct_kernel_pruning.py in fine_tuning_model/ directory

python unstruct_kernel_pruning.py --path=[weight file path to prune] --save=[directory path to save pruned weight]


##Fine-tuning
1. CIFAR-10 MobileNetV2
Perform fine-tuning in fine_tuned_model directory
The fine-tuning process is performed with unstructured kernel pruned weights

python main --resume=[weight file path to fine-tune] --fine_tuning --batch_size=[256] --lr=1e-2 --save=[directory path to save pruned weight]
--fine_tuning : Perform fine-tuning
--batch_size : Set batch size
--lr : Set learning rate

2. CIFAR-10 ResNet56
python main --resume=[weight file path to fine-tune] --fine_tuning --batch_size=[256] --lr=1e-2 --save=[directory path to save pruned weight] -a resnet56
--fine_tuning : Perform fine-tuning
--batch_size : Set batch size
--lr : Set learning rate
-a : specify the model

3. CIFAR-10 ResNet110
python main --resume=[weight file path to fine-tune] --fine_tuning --batch_size=[256] --lr=1e-2 --save=[directory path to save pruned weight] -a resnet110
--fine_tuning : Perform fine-tuning
--batch_size : Set batch size
--lr : Set learning rate
-a : specify the model

4. ImageNet MobileNetV2
python main --resume=[weight file path to fine-tune] --fine_tuning --batch_size=[256] --lr=1e-2 --save=[directory path to save pruned weight]
--fine_tuning : Perform fine-tuning
--batch_size : Set batch size
--lr : Set learning rate

5. ImageNet ResNet-50
python main --resume=[weight file path to fine-tune] --fine_tuning --batch_size=[256] --lr=1e-2 --save=[directory path to save pruned weight]
--fine_tuning : Perform fine-tuning
--batch_size : Set batch size
--lr : Set learning rate


##Structured Kernel Pruning
You can find unstruct2struct.py in fine_tuning_model/ directory
unstruct2struct.py

python unstruct2struct.py --path=[weight file path to structured kernel pruning] --save=[directory path to save]


##Compress Kernel Index Set
You can find encoder.py in channel_index_model/ directory
encoder.py

python encoder.py --path=[weight file path to compress kernel index set] --save=[directory path to save]

##Decode Kernel Index Set
You can find decoder.py in channel_index_model/ directory
decoder.py

python decoder.py --path=[weight file path to compress kernel index set] --save=[directory path to save]

##You can Check weight structure
check_weight.py

python check_weight.py --path=[wight file path to check structure]


##Parameter Counter
You can find parameter_counter.py in fine_tuning_model/ directory

python parameter_counter.py --path=[unstructured kernel pruned weight file path to count parameters]


##FLOPs Count
FLOPs are measured in a fine-tuning model.
1. CIFAR10 MobileNet V2

python main -e --resume=[weight file path to count flops] -a mobilenetv2_flops

2. CIFAR10 ResNet56

python main -e --resume=[weight file path to count flops] -a resnet56_flops

3. CIFAR10 ResNet110

python main -e --resume=[weight file path to count flops] -a resnet110_flops

4. ImageNet MobileNetV2

python main -e --resume=[weight file path to count flops] -a mobilenetv2_flops

5. ImageNet ResNet-50

python main -e --resume=[weight file path to count flops] -a resnet50_flops
