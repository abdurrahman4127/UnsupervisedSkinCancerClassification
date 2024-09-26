# MoCo (Momentum Contrast)

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

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from PIL import Image
import os
import torchvision.transforms as transforms
import copy

# the backbone model
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.output_dim = self.model.fc.in_features
        self.model.fc = nn.Identity()

    def forward(self, x):
        return self.model(x)

# MoCo
class MoCo(nn.Module):
    def __init__(self, backbone, dim=128, K=65536, m=0.999, T=0.07):
        super().__init__()
        self.dim = dim
        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = backbone
        self.encoder_k = copy.deepcopy(backbone)

        self.projection_q = nn.Sequential(
            nn.Linear(self.encoder_q.output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, dim)
        )
        self.projection_k = nn.Sequential(
            nn.Linear(self.encoder_k.output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, dim)
        )

        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        q = self.projection_q(self.encoder_q(im_q))
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.projection_k(self.encoder_k(im_k))
            k = nn.functional.normalize(k, dim=1)

        # positive logits: Bxl
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        # negative logits: BxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss = nn.CrossEntropyLoss()(logits / self.T, labels)

        self._dequeue_and_enqueue(k)

        return loss

class ImageDataset(nn.Module):
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
            q_image = self.transform(image)
            k_image = self.transform(image)

        return (q_image, k_image), 0

import torchvision.transforms as T

backbone = Backbone()

transforms = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = ImageDataset(dst_dir, transform=transforms)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

model = MoCo(backbone)
optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)

num_epochs = 5
for epoch in range(num_epochs):
    for (q_images, k_images), _ in train_loader:
        optimizer.zero_grad()
        loss = model(q_images, k_images)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}]')

def evaluate_representations(model, dataloader):
    model.eval()
    all_embeddings = []

    for inputs, _ in dataloader:
        embeddings = model.encoder_q(inputs[0])
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

print("MoCo Results:")
for k, metrics in results.items():
    print(f"for k = {k}, NMI: {metrics['NMI']:.4f}, ARI: {metrics['ARI']:.4f}")
