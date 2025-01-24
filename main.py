import argparse
import numpy as np
import torch
from server import FedRIRServer
from dataset.dataset import generate_dataset

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_trials", type=int, default=3, help="Number of trials")
    
    # Training parameters
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("--mask_ratio", type=float, default=0.6, help="Ratio of masked patches")
    parser.add_argument("--local_epochs", type=int, default=1, help="Number of local training epochs")
    parser.add_argument("--global_epochs", type=int, default=1000, help="Maximum number of global rounds")
    parser.add_argument("--eval_interval", type=int, default=1, help="Evaluation interval")
    
    # Federated learning parameters
    parser.add_argument("--join_ratio", type=float, default=1.0, help="Ratio of clients participating in each round")
    parser.add_argument("--random_join_ratio", type=bool, default=False, help="Whether to use random join ratio")
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset name', choices=['MNIST', 'Cifar10', 'Cifar100', 'FashionMNIST', 'OfficeCaltech10', 'DomainNet'])
    parser.add_argument('--base_data_dir', type=str, default='/data/HYQ/FedRIR/data', help='Base directory for datasets')
    parser.add_argument('--num_clients', type=int, default=20, help='Number of clients')
    parser.add_argument('--noniid', type=bool, default=True, help='Whether to use non-IID data distribution')
    parser.add_argument('--balance', type=bool, default=False, help='Whether to use balanced data distribution')
    parser.add_argument('--partition', type=str, default='dir', help='Data partition strategy (pat/dir/exdir)', choices=['pat', 'dir', 'exdir'])
    parser.add_argument('--alpha', type=float, default=0.1, help='Dirichlet distribution parameter')
    parser.add_argument('--train_ratio', type=float, default=0.75, help='Train/test split ratio')
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment_name = generate_dataset(args)
    args.experiment_name = experiment_name
    # real-world datasets setting
    if args.dataset == "DomainNet":
        args.num_clients = 6
    elif args.dataset == "OfficeCaltech10":
        args.num_clients = 4
    print(args)

    best_accs = []

    print("Training starts...")

    for i in range(args.num_trials):
        print(f"Trial {i+1} of {args.num_trials}")
        server = FedRIRServer(args)
        server.run()
        best_accs.append(server.best_test_acc)

    print(f"test accuracy: {np.mean(best_accs):.4f} ± {np.std(best_accs):.4f}")
    print(f"max test accuracy: {np.max(best_accs):.4f}")


if __name__ == "__main__":
    main()