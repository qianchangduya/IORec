import numpy as np
import torch
import scipy.sparse as sp
from scipy.spatial.distance import cosine, euclidean
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.stats import entropy
from data_utils import data_load, feature_load
import matplotlib.pyplot as plt

# 设置数据路径
data_path = '../data/meituan/'
train_path = data_path + 'iid/training_list.npy'
valid_path = data_path + 'iid/validation_dict.npy'
test_path = data_path + 'ood/testing_dict.npy'
user_feat_path = data_path + 'iid/user_feature.npy'
item_feat_path = data_path + 'iid/item_feature.npy'

# 加载数据
data_set = 'meituan'
train_data, valid_x_data, valid_y_data, test_x_data, test_y_data, n_users, n_items = \
    data_load(train_path, valid_path, test_path, data_set)

# 加载用户和物品特征
user_feature, item_feature = feature_load(user_feat_path, item_feat_path)


# 数据分类函数
def classify_users_by_shift(train_data, test_data, user_features):
    """
    根据数据偏移将用户分为IID和OOD类别。
    Args:
        train_data: IID交互数据 (稀疏矩阵)。
        test_data: OOD交互数据 (稀疏矩阵)。
        user_features: 用户特征矩阵。
    Returns:
        iid_users: 第一类用户（无偏移）。
        ood_users: 第二类用户（发生偏移）。
    """
    iid_users = []
    ood_users = []

    # 特征降维
    pca = PCA(n_components=2)  # 降维到二维便于可视化
    reduced_features = pca.fit_transform(user_features)

    # 聚类分析（尝试不同簇数）
    cluster_range = range(2, 6)
    results = []

    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(reduced_features)

        # 可视化：不同簇的用户
        plt.figure(figsize=(6, 4))
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, cmap='viridis')
        plt.colorbar(scatter)
        plt.title(f'K-Means Clustering with {n_clusters} Clusters')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.show()

        # 存储聚类结果用于后续分析
        results.append(cluster_labels)

    # 比较不同簇数下IID和OOD用户的分类效果
    # 假设已经有iid_users和ood_users的ID列表
    iid_users = np.array(iid_users)
    ood_users = np.array(ood_users)

    # 计算每个簇中IID和OOD用户的分布情况
    for i, n_clusters in enumerate(cluster_range):
        cluster_labels = results[i]

        iid_cluster_counts = [np.sum(cluster_labels[iid_users] == label) for label in range(n_clusters)]
        ood_cluster_counts = [np.sum(cluster_labels[ood_users] == label) for label in range(n_clusters)]

        plt.figure(figsize=(6, 4))
        plt.plot(range(n_clusters), iid_cluster_counts, label='IID Users', marker='o')
        plt.plot(range(n_clusters), ood_cluster_counts, label='OOD Users', marker='x')
        plt.title(f'IID and OOD Users Distribution for {n_clusters} Clusters')
        plt.xlabel('Cluster')
        plt.ylabel('User Count')
        plt.legend()
        plt.show()


# 执行用户分类
iid_users, ood_users = classify_users_by_shift(train_data, test_x_data, user_feature)

# 保存分类结果
np.save(data_path + 'Canshu_iid_users.npy', np.array(iid_users))
np.save(data_path + 'Canshu_ood_users.npy', np.array(ood_users))

# 打印分类结果
print(f'IID Users: {len(iid_users)}')
print(f'OOD Users: {len(ood_users)}')
