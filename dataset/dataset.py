import torch

from dataset.generate_MNIST import generate_MNIST
from dataset.generate_Cifar10 import generate_Cifar10
from dataset.generate_Cifar100 import generate_Cifar100
from dataset.generate_FashionMNIST import generate_FashionMNIST
from dataset.generate_OfficeCaltech10 import generate_OfficeCaltech10
from dataset.generate_DomainNet import generate_DomainNet


torch.manual_seed(0)

def generate_dataset(args):
    if args.dataset == "MNIST":
        experiment_name = generate_MNIST(args)
    elif args.dataset == "Cifar10":
        experiment_name = generate_Cifar10(args)
    elif args.dataset == "Cifar100":
        experiment_name = generate_Cifar100(args)
    elif args.dataset == "FashionMNIST":
        experiment_name = generate_FashionMNIST(args)
    elif args.dataset == "DomainNet":
        experiment_name = generate_DomainNet(args)
    elif args.dataset == "OfficeCaltech10":
        experiment_name = generate_OfficeCaltech10(args)
    else:
        raise NotImplementedError("unexpected dataset: {}".format(args.dataset))
    return experiment_name