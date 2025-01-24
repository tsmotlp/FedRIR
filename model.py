import torch
import torch.nn as nn


# Global Feature Extractor
class GFE(nn.Module):
    def __init__(self, in_channels=1, hidden_size=1024, embed_dim=512):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc = nn.Linear(hidden_size, embed_dim)
    
    def forward(self, x):
        z = self.convs(x)
        z = z.view(x.size(0), -1)
        z = self.fc(z)
        return z


# Client-Specific Feature Extractor
class CSFE(nn.Module):
    def __init__(self, in_channels=1, hidden_size=1024, embed_dim=512):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc = nn.Linear(hidden_size, embed_dim * 2)
    
    def forward(self, x):
        z = self.convs(x)
        z = z.view(x.size(0), -1)
        z = self.fc(z)
        return z


class Generator(nn.Module):
    def __init__(self, in_channels=1, hidden_size=1024, embed_dim=512):
        super().__init__()
        self.patch_size = int((hidden_size // 64) ** .5)
        self.generator_fc = nn.Linear(in_features=embed_dim, out_features=hidden_size)
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=0, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=5, stride=2, padding=0, output_padding=1),
        )
    
    def forward(self, z):
        z = self.generator_fc(z)
        z = z.view(z.size(0), -1, self.patch_size, self.patch_size)
        output = self.generator(z)
        return output

# Information Distillation Module
class IDM(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mu = nn.Sequential(nn.Linear(embed_dim, 128),
                                nn.ReLU(),
                                nn.Linear(128, embed_dim))

        self.logvar = nn.Sequential(nn.Linear(embed_dim, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, embed_dim),
                                    nn.Tanh())

    def forward(self, gfe, csfe):
        mu = self.mu(gfe)
        logvar = self.logvar(gfe)
        return -1.0 * (-(mu - csfe)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)


class FedRIRModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, hidden_size=1024, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.gfe = GFE(in_channels, hidden_size, embed_dim)
        self.csfe = CSFE(in_channels, hidden_size, embed_dim)
        self.generator = Generator(in_channels, hidden_size, embed_dim)
        self.idm = IDM(embed_dim)
        self.phead = nn.Linear(embed_dim * 2, num_classes)
    
    def reconstruction(self, x, patch_size=4, mask_ratio=0.6):
        masked_x = self.mask_image(x, patch_size=patch_size, mask_ratio=mask_ratio)
        z = self.csfe(masked_x)
        mean, logvar = torch.split(z, self.embed_dim, dim=1)
        z = self.reparameterize(mean, logvar, training=True)
        recon = self.generator(z)
        return recon, mean, logvar

    def classification(self, x):
        # Freeze the CSFE
        with torch.no_grad():
            z = self.csfe(x)
            mean, logvar = torch.split(z, self.embed_dim, dim=1)
            csf = self.reparameterize(mean, logvar, training=False)
        gf = self.gfe(x)
        pf = torch.concat([gf, csf], dim=1)
        logits = self.phead(pf)
        return logits, gf, csf
    
    @staticmethod
    def reparameterize(mean, logvar, training=True):
        if training:
            std = torch.exp(logvar / 2)
            epsilon = torch.randn_like(std)
            return epsilon * std + mean
        else:
            return mean
    
    @staticmethod
    def mask_image(img, patch_size=4, mask_ratio=0.6):
        b, c, h, w = img.shape
        patch_h = patch_w = patch_size
        num_patches = (h // patch_h) * (w // patch_w)

        patches = img.view(
            b, c,
            h // patch_h, patch_h, 
            w // patch_w, patch_w
        ).permute(0, 2, 4, 3, 5, 1).reshape(b, num_patches, -1)
        
        num_masked = int(mask_ratio * num_patches)
        shuffle_indices = torch.rand(b, num_patches).argsort()
        mask_ind, unmask_ind = shuffle_indices[:, :num_masked], shuffle_indices[:, num_masked:]
        batch_ind = torch.arange(b).unsqueeze(-1)
        # masked
        patches[batch_ind, mask_ind] = 0
        x_masked = patches.view(
            b, h // patch_h, w // patch_w, 
            patch_h, patch_w, c
        ).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)
        
        return x_masked
    

def create_model(dataset):
    if dataset == "MNIST":
        return FedRIRModel(1, 10, 1024, 512)
    elif dataset == "FashionMNIST":
        return FedRIRModel(1, 10, 1024, 512)
    elif dataset == "Cifar10":
        return FedRIRModel(3, 10, 1600, 512)
    elif dataset == "Cifar100":
        return FedRIRModel(3, 100, 1600, 512)
    elif dataset == "OfficeCaltech10":
        return FedRIRModel(3, 10, 10816, 512)
    elif dataset == "DomainNet":
        return FedRIRModel(3, 10, 10816, 512)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
