import os
import cv2
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score

image_dir = '/content/dataset'

def load_and_preprocess_images(directory, color_space='HSV', bins=(8, 8, 8)):
    images = []

    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            img = cv2.imread(os.path.join(directory, filename))
            if color_space == 'HSV':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 128))
            images.append(img)

    return images

def extract_color_histograms(images, bins=(8, 8, 8)):
    histograms = []

    for img in images:
        hist = cv2.calcHist([img], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        histograms.append(hist)

    return histograms

def experiment(color_space, bins, pca, tsne):
    image_data = load_and_preprocess_images(image_dir, color_space, bins)
    random.shuffle(image_data)

    histograms = extract_color_histograms(image_data, bins)

    if pca:
        scaler = StandardScaler()
        histograms = scaler.fit_transform(histograms)

        pca = PCA(n_components=2)
        histograms_pca = pca.fit_transform(histograms)
    else:
        histograms_pca = histograms

    if tsne:
        if pca:
            histograms_tsne = histograms_pca
        else:
            scaler = StandardScaler()
            histograms_tsne = scaler.fit_transform(histograms)
            tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
            histograms_tsne = tsne.fit_transform(histograms)

    isic_dbi_scores = []
    isic_silhouette_scores = []

    num_clusters = list(range(1, 11))

    for k in num_clusters:
        # ISIC
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(histograms)
        dbi_avg = davies_bouldin_score(histograms, cluster_labels)
        isic_dbi_scores.append(dbi_avg)
        silhouette_avg = silhouette_score(histograms, cluster_labels)
        isic_silhouette_scores.append(silhouette_avg)

    plt.figure(figsize=(12, 6))
    plt.plot(num_clusters, isic_dbi_scores, marker='o', linestyle='-', color='blue', label='ISIC')
    plt.xticks(num_clusters)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Davies-Bouldin Index')
    plt.title('Davies-Bouldin Index Comparison')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(num_clusters, isic_silhouette_scores, marker='o', linestyle='-', color='blue', label='ISIC')
    plt.xticks(num_clusters)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score Comparison')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Display a few example images
    plt.figure(figsize=(12, 6))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(image_data[i])
        plt.axis('off')
    plt.show()

    # Plot a few color histograms
    num_histograms = 12
    num_rows = 3
    num_cols = 4

    random.shuffle(histograms)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(16, 12))

    for i, hist in enumerate(histograms[:num_histograms]):
        row, col = divmod(i, num_cols)
        axs[row, col].plot(hist)
        axs[row, col].set_title(f'Hist {i + 1}')
        axs[row, col].set_xlabel('Bin')
        axs[row, col].set_ylabel('Frequency')

    for i in range(num_histograms, num_rows * num_cols):
        fig.delaxes(axs[divmod(i, num_cols)])

    plt.tight_layout()
    plt.show()

    if pca:
        plt.figure(figsize=(8, 6))
        plt.scatter(histograms_pca[:, 0], histograms_pca[:, 1])
        plt.title('PCA Visualization')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.show()

    if tsne:
        plt.figure(figsize=(8, 6))
        plt.scatter(histograms_tsne[:, 0], histograms_tsne[:, 1])
        plt.title('t-SNE Visualization')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.show()

# #  the experiments
# experiment(color_space='HSV', bins=(8, 8, 8), pca=False, tsne=False)  # Exp1
# experiment(color_space='HSV', bins=(8, 8, 8), pca=False, tsne=True)   # Exp2
# experiment(color_space='HSV', bins=(8, 8, 8), pca=True, tsne=False)   # Exp3
# experiment(color_space='HSV', bins=(8, 8, 8), pca=True, tsne=True)    # Exp4
# experiment(color_space='RGB', bins=(8, 8, 8), pca=False, tsne=False)  # Exp5
# experiment(color_space='RGB', bins=(8, 8, 8), pca=False, tsne=True)   # Exp6
# experiment(color_space='RGB', bins=(8, 8, 8), pca=True, tsne=False)   # Exp7
experiment(color_space='RGB', bins=(8, 8, 8), pca=True, tsne=True)    # Exp8
# experiment(color_space='HSV', bins=(16, 16, 16), pca=False, tsne=False)  # Exp9
# experiment(color_space='HSV', bins=(16, 16, 16), pca=False, tsne=True)   # Exp10
# experiment(color_space='HSV', bins=(16, 16, 16), pca=True, tsne=False)   # Exp11
# experiment(color_space='HSV', bins=(16, 16, 16), pca=True, tsne=True)    # Exp12
# experiment(color_space='RGB', bins=(16, 16, 16), pca=False, tsne=False)  # Exp13
# experiment(color_space='RGB', bins=(16, 16, 16), pca=False, tsne=True)   # Exp14
# experiment(color_space='RGB', bins=(16, 16, 16), pca=True, tsne=False)   # Exp15
# experiment(color_space='RGB', bins=(16, 16, 16), pca=True, tsne=True)    # Exp16 
