import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

n_class = 80
n_sample = 500
n_feature_dim = 512
n_cluster = 20

feature_npy = 'X_train_fv_all.npy'
feature_np = np.load(feature_npy).reshape(n_class, n_sample*2, n_feature_dim)
# feature_np = np.random.rand(n_class, n_sample, n_feature_dim)
classes_centroids = []
for i in tqdm(range(n_class)):
    class_i_feature_np = feature_np[i]
    kmeans = KMeans(n_clusters=n_cluster, random_state=2).fit(class_i_feature_np)
    classes_centroids.append(kmeans.cluster_centers_)

np.save('classes_centroids.npy', np.array(classes_centroids))