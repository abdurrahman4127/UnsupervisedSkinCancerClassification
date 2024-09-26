# Advancing Skin Cancer Detection Integrating a Novel Unsupervised Classification and Enhanced Imaging Techniques

This repository contains the implementation code and pipeline for our novel unsupervised learning methodology for skin lesion classification. 

## Methodology
Our methodology combines modified ESRGAN, a novel histogram feature extraction map, optimal cluster-number estimation, and the application of an unsupervised clustering algorithm.

![Methodology](/images/methodology.png)

## Dataset
The following two public datasets were used in our experiment:
- **ISIC 2019**: [andrewmvd/isic‐2019](https://www.kaggle.com/datasets/andrewmvd/isic‐2019)
- **HAM10000**: [kmader/skin-cancer-mnist-ham10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

## ESRGAN

The detailed implementation of ESR‐Generative Adversarial Networks is available at [Code 📁](https://github.com/abdurrahman4127/UnsupervisedSkinCancerClassification/tree/main/ESRGAN), and the other experimented GAN-based image enhancer pretrained models can be downloaded using scripts at [.sh 📁](https://github.com/abdurrahman4127/UnsupervisedSkinCancerClassification/tree/main/Pretrained%20GAN-based%20Models).

##### ESRGAN Output:
![esrgan_op](/images/esrgan_output.png)


## Histogram Feature Map
Histogram feature map generation and extraction details are available at [Code 📁](https://github.com/abdurrahman4127/UnsupervisedSkinCancerClassification/blob/main/Unsupervised_HistogramFreature.ipynb)

##### Feature Map Visualization:
![hist_feature](/images/histogram.png)
![hist_feature2](/images/histogram2.png)

## Optimal Cluster Number
Experimented optimal number of cluster findings with DBI and SS score evaluation code is detailed at [Code 📁](https://github.com/abdurrahman4127/UnsupervisedSkinCancerClassification/blob/main/Unsupervised_HistogramFreature.ipynb)

##### Visualization of Optimal Cluster Number:
![optimal_cluster_num](/images/cluster_num.png)

## Clustering Performance Evaluation
k-Means clustering algorithm was finally chosen for our approach after exploring several other clustering algorithms. Code is available at [Code 📁](https://github.com/abdurrahman4127/UnsupervisedSkinCancerClassification/blob/main/Unsupervised_HistogramFreature.ipynb)

##### Evaluation of k-Means Clustering Approach:
![kmeans](/images/kmeans.png)

## Requirements

The following key Python packages are required to run the code:

- TensorFlow
- PyTorch
- NumPy
- Keras
- Pandas
- Matplotlib


## Acceptance
The paper is accepted in CAAI Transactions on Intelligence Technology on 26th Sept 2024. 

## Citation
Will be updated soon. 
