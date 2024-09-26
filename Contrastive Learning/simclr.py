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

# the backbone model
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.output_dim = self.model.fc.in_features
        self.model.fc = nn.Identity()

    def forward(self, x):
        return self.model(x)

# SimCLR
class SimCLR(nn.Module):
    def __init__(self, backbone, projection_dim=128):
        super().__init__()
        self.backbone = backbone
        self.projection_head = nn.Sequential(
            nn.Linear(backbone.output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        z = self.backbone(x)
        p = self.projection_head(z)
        return p

    def training_step(self, batch):
        (x1, x2), _ = batch
        z1 = self.forward(x1)
        z2 = self.forward(x2)
        loss = self.contrastive_loss(z1, z2)
        return loss

    def contrastive_loss(self, z1, z2, temperature=0.1):
        N, _ = z1.shape
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        similarity_matrix = torch.matmul(z1, z2.T)
        log_prob = torch.log(torch.exp(similarity_matrix / temperature) / torch.sum(torch.exp(similarity_matrix / temperature), dim=1, keepdim=True))
        loss = -torch.mean(torch.diag(log_prob))
        return loss

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
            image1 = self.transform(image)
            image2 = self.transform(image)
        return (image1, image2), 0

backbone = Backbone()

transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = ImageDataset(dst_dir, transform=transforms)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Train the model
model = SimCLR(backbone)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# trying out with 5 epochs
num_epochs = 5

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        loss = model.training_step(batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}]')

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

def evaluate_representations(model, dataloader):
    model.eval()
    all_embeddings = []

    for inputs, _ in dataloader:
        embeddings = model.backbone(inputs[0])
        all_embeddings.extend(embeddings.detach().cpu().numpy())

    results = {}
    for num_clusters in range(6, 11):
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(all_embeddings)
        cluster_labels = kmeans.labels_

        # Calculate NMI and ARI
        nmi = normalized_mutual_info_score(range(len(all_embeddings)), cluster_labels)
        ari = adjusted_rand_score(range(len(all_embeddings)), cluster_labels)

        results[num_clusters] = {
            'NMI': nmi,
            'ARI': ari
        }

    return results

results = evaluate_representations(model, test_loader)

print("SimCLR Results:")
for k, metrics in results.items():
    print(f"for k = {k}, NMI: {metrics['NMI']:.4f}, ARI: {metrics['ARI']:.4f}")
