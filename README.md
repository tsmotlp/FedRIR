# FedRIR

This is the official PyTorch implementation of our WWW'25 **Oral** paper:

**FedRIR: Rethinking Information Representation in Federated Learning**

## Overview
![Overview](assets/overview.png)

## Requirements
- Python 3.x
- PyTorch
- torchvision
- numpy

## Datasets
We conduct experiments on six datasets:
- MNIST
- CIFAR-10
- CIFAR-100
- FashionMNIST 
- OfficeCaltech10
- DomainNet

## Training
```
python main.py --dataset MNIST --num_clients 20 --global_epochs 1000 --join_ratio 1.0 --partition dir --alpha 0.1 --train_ratio 0.75
```

## Citation
If you find this code useful for your research, please cite our paper:
```
[Citation will be available after publication]
```

## License
This project is licensed under the MIT License.

## Contact
If you have any questions, please feel free to contact [your email].
