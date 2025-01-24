import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.utils import read_client_data
from model import create_model

class FedRIRClient:
    def __init__(self, args, client_id):
        self.experiment_name = args.experiment_name
        self.client_id = client_id
        self.device = args.device
        self.base_data_dir = args.base_data_dir
        self.dataset = args.dataset
        self.train_samples = 0
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.mask_ratio = args.mask_ratio

        self.model = create_model(args.dataset).to(args.device)
        self.gfe = copy.deepcopy(self.model.gfe)

        self.recon_loss = nn.MSELoss().to(self.device)
        self.cls_loss = nn.CrossEntropyLoss().to(self.device)
        
        self.opt_reconstruction = torch.optim.Adam(
            list(self.model.csfe.parameters()) + list(self.model.generator.parameters()),
            lr=args.lr
        )
        self.opt_classification = torch.optim.Adam(
            list(self.model.gfe.parameters()) + list(self.model.idm.parameters()) + list(self.model.phead.parameters()),
            lr=args.lr
        )

        self.train_loader = self.load_train_data()
        self.test_loader = self.load_test_data()

    def set_model(self, gfe):
        for new_param, old_param in zip(gfe.parameters(), self.model.gfe.parameters()):
            old_param.data = new_param.data.clone()
        for new_param, old_param in zip(gfe.parameters(), self.gfe.parameters()):
            old_param.data = new_param.data.clone()


    def test_metrics(self):
        self.model.eval()

        test_corrects = 0
        test_cls_loss = 0
        test_num_samples = 0

        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits, _, _ = self.model.classification(x)
                test_corrects += (torch.sum(torch.argmax(logits, dim=1) == y)).item()    
                test_cls_loss += self.cls_loss(logits, y).item() * y.shape[0]
                test_num_samples += y.shape[0]
        return {
            "test_num_samples": test_num_samples,
            "test_corrects": test_corrects,
            "test_cls_loss": test_cls_loss
        }
    
    def train_metrics(self):
        self.model.eval()

        train_num_samples = 0
        train_cls_loss = 0
        train_id_loss = 0
        train_recon_loss = 0
        train_corrects = 0
        with torch.no_grad():
            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                recon, mean, logvar = self.model.reconstruction(x, mask_ratio=self.mask_ratio)
                recon_loss = self.recon_loss(recon, x) + calc_kld_loss(mean, logvar) * 0.1
                logits, gf, csf = self.model.classification(x)      
                cls_loss = self.cls_loss(logits, y)
                id_loss = self.model.idm(gf, csf)
                train_num_samples += y.shape[0]
                train_cls_loss += cls_loss.item() * y.shape[0]
                train_id_loss += id_loss.item() * y.shape[0]
                train_recon_loss += recon_loss.item() * y.shape[0]
                train_corrects += (torch.sum(torch.argmax(logits, dim=1) == y)).item()

        return {
            "train_num_samples": train_num_samples,
            "train_recon_loss": train_recon_loss,
            "train_cls_loss": train_cls_loss,
            "train_id_loss": train_id_loss,
            "train_corrects": train_corrects
        }

                
    def train_reconstruction(self):
        self.model.train()

        for epoch in range(self.local_epochs):
            for i, (x, y) in enumerate(self.train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                self.opt_reconstruction.zero_grad()
                recon, mean, logvar = self.model.reconstruction(x, mask_ratio=self.mask_ratio)
                recon_loss = self.recon_loss(recon, x) + calc_kld_loss(mean, logvar) * 0.1
                recon_loss.backward()
                self.opt_reconstruction.step()

        torch.cuda.empty_cache()

    def train_classification(self):
        self.model.train()

        for epoch in range(self.local_epochs):
            for i, (x, y) in enumerate(self.train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                self.opt_classification.zero_grad()
                logits, gf, csf = self.model.classification(x)
                cls_loss = self.cls_loss(logits, y)
                id_loss = self.model.idm(gf, csf)
                loss = cls_loss + id_loss
                loss.backward()
                self.opt_classification.step()
        
        torch.cuda.empty_cache()

    def train(self):
        # Stage 1: Train reconstruction
        self.train_reconstruction()
        
        # Stage 2: Train classification
        self.train_classification()

    def load_train_data(self):
        train_data = read_client_data(self.base_data_dir, self.dataset, self.experiment_name, self.client_id, is_train=True)
        self.train_samples = len(train_data)
        return DataLoader(train_data, self.batch_size, drop_last=True, shuffle=True)

    def load_test_data(self):
        test_data = read_client_data(self.base_data_dir, self.dataset, self.experiment_name, self.client_id, is_train=False)
        return DataLoader(test_data, self.batch_size, drop_last=False, shuffle=True)


def calc_kld_loss(mu, log_var):
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    return kld_loss