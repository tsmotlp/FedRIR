# FedRIR

This is the official PyTorch implementation of our WWW'25 **Oral** paper:

**FedRIR: Rethinking Information Representation in Federated Learning**

## Overview
<p align="center">
  <img src="assets/overview.png" width="50%" alt="Overview">
</p>

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
@inproceedings{huang2025fedrir,
  title={FedRIR: Rethinking Information Representation in Federated Learning},
  author={Huang, Yongqiang and Shao, Zerui and Yang, Ziyuan and Lu, Zexin and Zhang, Yi},
  booktitle={Proceedings of the ACM on Web Conference 2025},
  pages={807--816},
  year={2025}
}
```

## License
This project is licensed under the MIT License.

## Contact
If you have any questions, please feel free to contact [yqhuang2912@gmail.com].
