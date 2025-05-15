import numpy as np
from data_utils import data_load, feature_load
import scipy.sparse as sp  # 导入SciPy用于稀疏矩阵的支持

# 数据路径
data_path = '../data/meituan002/'
train_path = data_path + 'iid/training_list.npy'
valid_path = data_path + 'iid/validation_dict.npy'
test_path = data_path + 'ood/testing_dict.npy'
user_feat_path = data_path + 'iid/user_feature.npy'
ood_train_path = data_path + 'ood/training_list.npy'
ood_user_feat_path = data_path + 'ood/user_feature.npy'

# 加载数据
train_list = np.load(train_path, allow_pickle=True)  # 加载训练数据列表
ood_train_list = np.load(ood_train_path, allow_pickle=True)  # 加载训练数据列表
valid_dict = np.load(valid_path, allow_pickle=True).item()  # 加载验证数据字典
test_dict = np.load(test_path, allow_pickle=True).item()  # 加载测试数据字典

# 构建训练字典
uid_max = 0  # 用户ID最大值
iid_max = 0  # 物品ID最大值
train_dict = {}  # 初始化训练字典
for entry in train_list:  # 遍历训练数据列表
    user, item = entry  # 获取用户和物品
    if user not in train_dict:  # 如果用户不在字典中
        train_dict[user] = []  # 初始化用户对应的物品列表
    train_dict[user].append(item)  # 将物品添加到用户的物品列表中
    if user > uid_max:  # 更新用户ID最大值
        uid_max = user
    if item > iid_max:  # 更新物品ID最大值
        iid_max = item

# 构建验证列表和测试列表
valid_list = []  # 初始化验证列表
test_list = []  # 初始化测试列表
for u in valid_dict:  # 遍历验证字典
    if u > uid_max:  # 更新用户ID最大值
        uid_max = u
    for i in valid_dict[u]:  # 遍历用户对应的验证物品
        valid_list.append([u, i])  # 将用户和物品对添加到验证列表中
        if i > iid_max:  # 更新物品ID最大值
            iid_max = i

for u in test_dict:  # 遍历测试字典
    if u > uid_max:  # 更新用户ID最大值
        uid_max = u
    for i in test_dict[u]:  # 遍历用户对应的测试物品
        test_list.append([u, i])  # 将用户和物品对添加到测试列表中
        if i > iid_max:  # 更新物品ID最大值
            iid_max = i

n_users = max(uid_max + 1, 25768)
n_items = max(iid_max + 1, 3044)
print(f'n_users: {n_users}')  # 打印用户数量
print(f'n_items: {n_items}')  # 打印物品数量

# 构建训练数据的稀疏矩阵
train_data = sp.csr_matrix((np.ones_like(train_list[:, 0]),
                            (train_list[:, 0], train_list[:, 1])), dtype='float64',
                           shape=(n_users, n_items))
ood_train_data = sp.csr_matrix((np.ones_like(ood_train_list[:, 0]),
                            (ood_train_list[:, 0], ood_train_list[:, 1])), dtype='float64',
                           shape=(n_users, n_items))

# 加载用户和物品特征
iid_user_feature = np.load(user_feat_path, allow_pickle=True)  # 加载用户特征
ood_user_feature = np.load(ood_user_feat_path, allow_pickle=True)  # 加载用户特征


# ======== 1. 用户特征变化计算 ========
# 计算标准化欧几里得距离和特征差异均值
def calculate_user_feature_change(iid_user_feature, ood_user_feature):
    # 标准化用户特征
    iid_user_feature_norm = (iid_user_feature - np.mean(iid_user_feature, axis=0)) / (
                np.std(iid_user_feature, axis=0) + 1e-8)
    ood_user_feature_norm = (ood_user_feature - np.mean(ood_user_feature, axis=0)) / (
                np.std(ood_user_feature, axis=0) + 1e-8)

    # 标准化欧几里得距离
    euclidean_distance = np.linalg.norm(iid_user_feature_norm - ood_user_feature_norm, axis=1)

    # 特征差异的绝对值均值
    feature_difference = np.mean(np.abs(iid_user_feature - ood_user_feature), axis=1)

    return euclidean_distance, feature_difference


user_feature_euclidean, user_feature_difference = calculate_user_feature_change(iid_user_feature, ood_user_feature)


# ======== 2. 用户-物品交互变化计算 ========
# 使用 Jaccard 相似度衡量用户交互变化
def calculate_interaction_change(iid_matrix, ood_matrix):
    interaction_overlap = np.sum(np.minimum(iid_matrix, ood_matrix), axis=1)
    interaction_union = np.sum(np.maximum(iid_matrix, ood_matrix), axis=1)
    jaccard_similarity = interaction_overlap / (interaction_union + 1e-8)

    # 交互行为差异均值
    interaction_difference = np.mean(np.abs(iid_matrix - ood_matrix), axis=1)

    return jaccard_similarity, interaction_difference


interaction_jaccard_sim, interaction_difference = calculate_interaction_change(
    train_data.toarray(), ood_train_data.toarray()
)


# ======== 3. 用户分类 ========
# 分类规则：根据用户特征变化和交互变化
def classify_users(euclidean_change, feature_diff, jaccard_sim, interaction_diff, thresholds):
    feature_threshold, interaction_threshold = thresholds
    iid_users = (euclidean_change <= feature_threshold[0]) & \
                (feature_diff <= feature_threshold[1]) & \
                (jaccard_sim >= interaction_threshold[0]) & \
                (interaction_diff <= interaction_threshold[1])
    ood_users = ~iid_users
    return iid_users, ood_users


# 阈值选择（动态选择基于分位数）
feature_threshold = (np.percentile(user_feature_euclidean, 40),  # 欧几里得距离中值
                     np.percentile(user_feature_difference, 50))  # 特征差异均值中值
interaction_threshold = (np.percentile(interaction_jaccard_sim, 30),  # Jaccard 相似度中值
                         np.percentile(interaction_difference, 50))  # 交互差异中值

iid_users, ood_users = classify_users(user_feature_euclidean, user_feature_difference,
                                      interaction_jaccard_sim, interaction_difference,
                                      (feature_threshold, interaction_threshold))


# 保存分类结果
np.save(data_path + 'demo6_iid_users.npy', np.array(iid_users))
np.save(data_path + 'demo6_ood_users.npy', np.array(ood_users))

# ======== 5. 结果输出 ========
print(f"Number of IID Users: {np.sum(iid_users)}")
print(f"Number of OOD Users: {np.sum(ood_users)}")


