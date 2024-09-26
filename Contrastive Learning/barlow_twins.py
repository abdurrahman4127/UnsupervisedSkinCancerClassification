# Barlow Twins

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers, Model
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, PReLU,BatchNormalization, Flatten
from keras.layers import UpSampling2D, LeakyReLU, Dense, Input, add
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, Normalize
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, normalized_mutual_info_score, adjusted_rand_score

import os
import shutil
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize, Resize, ToTensor

dst_dir = "/content/dataset"

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = os.listdir(root_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.files[idx])
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image

transforms = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

dataset = ImageDataset(dst_dir, transform=transforms)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, normalized_mutual_info_score, adjusted_rand_score
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from PIL import Image
import os
import torchvision.transforms as transforms
import copy

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from PIL import Image
import os
import torchvision.transforms as transforms

# the backbone model
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.output_dim = self.model.fc.in_features
        self.model.fc = nn.Identity()

    def forward(self, x):
        return self.model(x)

# Barlow Twins model
class BarlowTwins(nn.Module):
    def __init__(self, backbone, output_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            backbone,
            nn.Linear(backbone.output_dim, output_dim)
        )
        self.bn = nn.BatchNorm1d(output_dim, affine=False)

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        z1 = self.bn(z1)
        z2 = self.bn(z2)

        z1_norm = (z1 - z1.mean(dim=0)) / z1.std(dim=0)
        z2_norm = (z2 - z2.mean(dim=0)) / z2.std(dim=0)

        cross_correlation = torch.einsum('ij,kj->ik', [z1_norm, z2_norm]) / x1.size(0)

        on_diagonal = torch.diagonal(cross_correlation).add(-1).pow(2).mean()
        off_diagonal = cross_correlation.mul(torch.eye(cross_correlation.size(0), device=cross_correlation.device)).abs_().sum().sub(on_diagonal)
        loss = on_diagonal + 0.0051 * off_diagonal

        return loss

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = os.listdir(root_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.files[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image1 = self.transform(image)
            image2 = self.transform(image)

        return image1, image2

transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = ImageDataset(dst_dir, transform=transforms)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Train the model
model = BarlowTwins(Backbone())
optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)

num_epochs = 5
for epoch in range(num_epochs):
    for (images1, images2) in train_loader:
        optimizer.zero_grad()
        loss = model(images1, images2)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}]')

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

def evaluate_representations(model, dataloader):
    model.eval()
    all_embeddings = []

    for images1, images2 in dataloader:
        embeddings = model.encoder(images1)
        all_embeddings.extend(embeddings.detach().cpu().numpy())

    results = {}
    for num_clusters in range(6, 11):
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(all_embeddings)
        cluster_labels = kmeans.labels_

        nmi = normalized_mutual_info_score(range(len(all_embeddings)), cluster_labels)
        ari = adjusted_rand_score(range(len(all_embeddings)), cluster_labels)

        results[num_clusters] = {
            'NMI': nmi,
            'ARI': ari
        }

    return results

results = evaluate_representations(model, train_loader)

print("Barlow Twins Results:")
for k, metrics in results.items():
    print(f"for k = {k}, NMI: {metrics['NMI']:.4f}, ARI: {metrics['ARI']:.4f}")
