# Cluster-based Similarity Partitioning Algorithm (CSPA)
from sklearn.cluster import AgglomerativeClustering

# ISIC2019 dataset
n_clusters = 6
cspa = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='complete')
isic_cluster_labels = cspa.fit_predict(isic_histograms)

# HAM10000 dataset 
n_clusters = 5
cspa = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='complete')
ham10000_cluster_labels = cspa.fit_predict(ham10000_histograms)

save_cluster_assignments(isic_cluster_labels, 'isic_cspa_cluster_assignments.csv')
save_cluster_assignments(ham10000_cluster_labels, 'ham10000_cspa_cluster_assignments.csv')




# Invariant Information Clustering (IIC)
from iic.iic import IIC

# ISIC2019 dataset
n_clusters = 6
iic = IIC(n_clusters=n_clusters)
isic_cluster_labels = iic.fit_predict(isic_histograms)

# HAM10000 dataset
n_clusters = 5 
iic = IIC(n_clusters=n_clusters)
ham10000_cluster_labels = iic.fit_predict(ham10000_histograms)

save_cluster_assignments(isic_cluster_labels, 'isic_iic_cluster_assignments.csv')
save_cluster_assignments(ham10000_cluster_labels, 'ham10000_iic_cluster_assignments.csv')



# Gaussian Mixture Models (GMM)
from sklearn.mixture import GaussianMixture

# ISIC2019 dataset
n_clusters = 6
gmm = GaussianMixture(n_components=n_clusters)
isic_cluster_labels = gmm.fit_predict(isic_histograms)

# HAM10000 dataset
n_clusters = 5
gmm = GaussianMixture(n_components=n_clusters)
ham10000_cluster_labels = gmm.fit_predict(ham10000_histograms)

save_cluster_assignments(isic_cluster_labels, 'isic_gmm_cluster_assignments.csv')
save_cluster_assignments(ham10000_cluster_labels, 'ham10000_gmm_cluster_assignments.csv')
