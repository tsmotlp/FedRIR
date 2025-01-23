import numpy as np
import os
import torchvision.transforms as transforms
from os import path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from dataset.utils import check, prepare_path, separate_data, split_data, save_file


data_dir = "DomainNet"
 
# https://github.com/FengHZ/KD3A/blob/master/datasets/DomainNet.py
def read_domainnet_data(dataset_path, domain_name, split="train"):
    data_paths = []
    data_labels = []
    label_dict = {'bird':0, 'feather':1, 'headphones':2, 'ice_cream':3, 'teapot':4, 'tiger':5, 'whale':6, 'windmill':7, 'wine_glass':8, 'zebra':9}
    split_file = path.join(dataset_path, "splits", "{}_{}.txt".format(domain_name, split))
    with open(split_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data_path, label = line.split(' ')
            category = data_path.split("/")[1]
            print("data_path: ", data_path)
            print("data_path.split('/'): ", data_path.split('/'))
            data_path = path.join(dataset_path, data_path)
            print("data_path: ", data_path)
            print("category: ", category)
            if category in label_dict:
                label = label_dict[category]
                data_paths.append(data_path)
                data_labels.append(label)
    return data_paths, data_labels


class DomainNet(Dataset):
    def __init__(self, data_paths, data_labels, transforms, domain_name):
        super(DomainNet, self).__init__()
        self.data_paths = data_paths
        self.data_labels = data_labels
        self.transforms = transforms
        self.domain_name = domain_name

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        label = self.data_labels[index]
        img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.data_paths)


def get_domainnet_dloader(dataset_path, domain_name):
    train_data_paths, train_data_labels = read_domainnet_data(dataset_path, domain_name, split="train")
    test_data_paths, test_data_labels = read_domainnet_data(dataset_path, domain_name, split="test")
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.75, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    train_dataset = DomainNet(train_data_paths, train_data_labels, transforms_train, domain_name)
    train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_dataset = DomainNet(test_data_paths, test_data_labels, transforms_test, domain_name)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
    return train_loader, test_loader

# Allocate data to users
def generate_DomainNet(args):
    experiment_name, config_path, train_path, test_path = prepare_path(args.base_data_dir, data_dir, args)
    if check(config_path, train_path, test_path, args.num_clients, args.noniid, args.balance, args.partition, args.batch_size, args.alpha):
        return experiment_name
    
    root_dir = os.path.join(args.base_data_dir, data_dir, "rawdata")
    
    domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']

    urls = [
        'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip', 
        'http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip', 
        'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip', 
        'http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip', 
        'http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip', 
        'http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip', 
    ]
    http_head = 'http://csr.bu.edu/ftp/visda/2019/multi-source/'
    # Get DomainNet data
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
        if not os.path.exists(f"{root_dir}/splits"):
            os.makedirs(f"{root_dir}/splits")
            for d, u in zip(domains, urls):
                os.system(f'wget {u} -P {root_dir}')
                os.system(f'unzip {root_dir}/{d}.zip -d {root_dir}')
                print(f"{root_dir}/{d}.zip unzipped!")
                os.system(f'wget {http_head}domainnet/txt/{d}_train.txt -P {root_dir}/splits')
                os.system(f'wget {http_head}domainnet/txt/{d}_test.txt -P {root_dir}/splits')

    X, y = [], []
    for d in domains:
        train_loader, test_loader = get_domainnet_dloader(root_dir, d)

        for _, tt in enumerate(train_loader):
            train_data, train_label = tt
        for _, tt in enumerate(test_loader):
            test_data, test_label = tt

        dataset_image = []
        dataset_label = []

        dataset_image.extend(train_data.cpu().detach().numpy())
        dataset_image.extend(test_data.cpu().detach().numpy())
        dataset_label.extend(train_label.cpu().detach().numpy())
        dataset_label.extend(test_label.cpu().detach().numpy())

        X.append(np.array(dataset_image))
        y.append(np.array(dataset_label))

    labelss = []
    for yy in y:
        labelss.append(len(set(yy)))
    num_clients = len(y)
    print(f'Number of labels: {labelss}')
    print(f'Number of clients: {num_clients}')

    statistic = [[] for _ in range(num_clients)]
    for client in range(num_clients):
        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client]==i))))


    train_data, test_data = split_data(X, y)
    # modify the code in YOUR_ENV/lib/python3.8/site-packages/numpy/lib Line #678 from protocol=3 to protocol=4
    save_file(config_path, train_path, test_path, train_data, test_data, args.num_clients, max(labelss), 
        statistic, args.noniid, args.balance, args.partition, args.batch_size, args.alpha)
    return experiment_name