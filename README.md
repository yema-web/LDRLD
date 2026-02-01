# Local Dense Logit Relations for Enhanced Knowledge Distillation (ICCV 2025)

[![Paper](https://img.shields.io/badge/Paper-ICCV2025-blue)](https://openaccess.thecvf.com/content/ICCV2025/html/Xu_Local_Dense_Logit_Relations_for_Enhanced_Knowledge_Distillation_ICCV_2025_paper.html) &nbsp;

This repository contains the official PyTorch implementation of the paper:

> **Local Dense Logit Relations for Enhanced Knowledge Distillation** \
> *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025*

## Installation

This repo was tested with Ubuntu 16.04.5 LTS, Python 3.5, PyTorch 0.4.0, and CUDA 9.0. But it should be runnable with recent PyTorch versions >=0.4.0. Reference CRD [github](https://github.com/HobbitLong/RepDistiller)

## Running
1. Fetch Pretrained Teacher Models
Download the pretrained teacher models to the save/models directory:
```
sh scripts/fetch_pretrained_teachers.sh
```
2. Run Distillation (LDRLD). 
To reproduce the results of LDRLD on CIFAR-100, use the following command.
Example: ResNet32x4 (Teacher) -> ResNet8x4 (Student)
```
CUDA_VISIBLE_DEVICES=1 python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ldrld --model_s resnet8x4 -r 1.0 -b 7.0 -a 10.5 --trial 1 --kd_T 4
```
  where the flags are explained as:

    --path_t: specify the path of the teacher model

    --model_s: specify the student model

    --distill: specify the distillation method

    -r: the weight of the cross-entropy loss 

    -a: the weight of the local dense logit relation distillation loss

    -b: the weight of non-target distillation loss

    --trial: specify the experimental id to differentiate between multiple runs
    
    --kd: the temperature coefficient

## Citation
If you find this repo useful for your research, please consider citing the paper:
```
@inproceedings{xu2025local,
  title={Local dense logit relations for enhanced knowledge distillation},
  author={Xu, Liuchi and Liu, Kang and Liu, Jinshuai and Wang, Lu and Xu, Lisheng and Cheng, Jun},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4539--4549},
  year={2025}
}

@inproceedings{tian2019crd,
  title={Contrastive Representation Distillation},
  author={Yonglong Tian and Dilip Krishnan and Phillip Isola},
  booktitle={International Conference on Learning Representations},
  year={2020}
}
```