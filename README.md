# Dep-L0
This repository contains the code for [Dep-L0: Improving L0-based Network Sparsification via Dependency Modeling](https://arxiv.org/abs/2107.00070).

![Demo](https://github.com/leo-yangli/dep-l0/blob/main/demo.png?raw=True)

## Requirments
	torch
	torchvision
	tensorboardX

## Usage

### VGG16 + C10
    forward: python train_vgg.py --net dep --num_classes 10 --lamba 1e-6  --gpu [ID]
    backward: python train_vgg.py --net dep --back_dep --num_classes 10 --lamba 1e-6 --gpu [ID]
    hc: python train_vgg.py --net hc  --num_classes 10 --lamba 5e-7 --gpu [ID]

### VGG16 + C100:
    forward: python train_vgg.py --net dep  --num_classes 100 --lamba 1e-6  --gpu [ID]
    backward: python train_vgg.py --net dep  --back_dep  --num_classes 100 --lamba 1e-6--gpu [ID]
    hc: python train_vgg.py --net hc  --num_classes 100 --lamba 5e-7 --gpu [ID]

### ResNet 56 + C10:
    forward: python train_resnet56.py --net dep  --num_classes 10 --lamba 5e-7  --gpu [ID]
    backward: python train_resnet56.py --net dep --back_dep  --num_classes 10 --lamba 1e-6 --gpu [ID]
    hc: python train_resnet56.py --net hc  --num_classes 10 --lamba 5e-7 --gpu [ID]

### ResNet 56 + C100:
    forward: python train_resnet56.py --net dep  --num_classes 100 --lamba 5e-7 --gpu [ID]
    backward: python train_resnet56.py --net dep --back_dep  --num_classes 100 --lamba 1e-6 --gpu [ID]
    hc: python train_resnet56.py --net hc  --num_classes 100 --lamba 5e-7 --gpu [ID]

### ResNet 50 + ImageNet
    forward: python train_resnet50.py [imagenet_dir] --net dep  --lamba 5e-9 --gpu_id [ID] 
    backward: python train_resnet50.py [imagenet_dir] --net dep --back_dep --lamba 5e-9 --gpu_id [ID] 
    hc: python train_resnet50.py [imagenet_dir] --net hc --lamba 5e-9 --gpu_id [ID] 

## Citation
If you found this code useful, please cite our paper.

    @inproceedings{depl02021,
      title={{Dep-L0}: Improving L0-based Network Sparsification via Dependency Modeling,
      author={Yang Li and Shihao Ji},
      booktitle={The European Conference on Machine Learning (ECML)},
      year={2021}
    }
    
