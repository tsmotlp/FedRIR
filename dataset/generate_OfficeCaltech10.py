import numpy as np
import os
import torchvision.transforms as transforms
from os import path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from dataset.utils import check, prepare_path, separate_data, split_data, save_file


data_dir = "OfficeCaltech10"

def read_officecaltech10_data(dataset_path, domain_name):
    data_paths = []
    data_labels = []
    label_dict={'back_pack':0, 'bike':1, 'calculator':2, 'headphones':3, 'keyboard':4, 'laptop_computer':5, 'monitor':6, 'mouse':7, 'mug':8, 'projector':9}
    domain_data_path = path.join(dataset_path, domain_name)
    for category in os.listdir(domain_data_path):
        category_data_path = path.join(domain_data_path, category)
        for filename in os.listdir(category_data_path):
            data_path = path.join(category_data_path, filename)
            data_paths.append(data_path)
            data_labels.append(label_dict[category])
    return data_paths, data_labels


class OfficeCaltech10Net(Dataset):
    def __init__(self, data_paths, data_labels, transforms, domain_name):
        super(OfficeCaltech10Net, self).__init__()
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


def get_officecaltech10_dloader(dataset_path, domain_name):
    data_paths, data_labels = read_officecaltech10_data(dataset_path, domain_name)
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.75, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    dataset = OfficeCaltech10Net(data_paths, data_labels, data_transforms, domain_name)
    dataloader = DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=False)
    return dataloader

# Allocate data to users
def generate_OfficeCaltech10(args):
    experiment_name, config_path, train_path, test_path = prepare_path(args.base_data_dir, data_dir, args)
    if check(config_path, train_path, test_path, args.num_clients, args.noniid, args.balance, args.partition, args.batch_size, args.alpha):
        return experiment_name
    
    root_dir = os.path.join(args.base_data_dir, data_dir, "rawdata")
    
    domains = ['amazon', 'caltech', 'dslr', 'webcam']

    X, y = [], []
    for d in domains:
        dataLoader = get_officecaltech10_dloader(root_dir, d)

        for _, tt in enumerate(dataLoader):
            data, label = tt

        dataset_image = []
        dataset_label = []

        dataset_image.extend(data.cpu().detach().numpy())
        dataset_label.extend(label.cpu().detach().numpy())

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