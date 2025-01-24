import numpy as np
import os
import torch
import torchvision
import torchvision.transforms as transforms
from dataset.utils import check, prepare_path, separate_data, split_data, save_file

data_dir = "Cifar10"


# Allocate data to users
def generate_Cifar10(args):
    experiment_name, config_path, train_path, test_path = prepare_path(args.base_data_dir, data_dir, args)
    if check(config_path, train_path, test_path, args.num_clients, args.noniid, args.balance, args.partition, args.batch_size, args.alpha):
        return experiment_name
    
    root_dir = os.path.join(args.base_data_dir, data_dir, "rawdata")
    # Get Cifar10 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(
        root=root_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root=root_dir, train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    X, y, statistic = separate_data((dataset_image, dataset_label), args.num_clients, num_classes, 
                                    args.noniid, args.balance, args.partition, args.batch_size, 
                                    args.train_ratio, args.alpha, class_per_client=2)
    train_data, test_data = split_data(X, y, args.train_ratio)
    save_file(config_path, train_path, test_path, train_data, test_data, args.num_clients, num_classes, 
        statistic, args.noniid, args.balance, args.partition, args.batch_size, args.alpha)
    return experiment_name