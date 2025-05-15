import numpy as np
import torch.utils.data as data  # 导入PyTorch数据处理相关模块
import scipy.sparse as sp  # 导入SciPy用于稀疏矩阵的支持

def feature_load(user_feat_path, item_feat_path):
    # 加载用户和物品特征
    user_feature = np.load(user_feat_path, allow_pickle=True)  # 加载用户特征
    item_feature = np.load(item_feat_path, allow_pickle=True)  # 加载物品特征
    return user_feature, item_feature  # 返回用户和物品特征


""" 构建VAE训练数据集 """
def data_load(train_path, valid_path, test_path, dataset=None):
    # 从指定路径加载训练、验证和测试数据
    train_list = np.load(train_path, allow_pickle=True)  # 加载训练数据列表
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


    # 根据数据集类型设置用户和物品数目上限
    if dataset == 'synthetic':
        n_users = max(uid_max + 1, 1000)
        n_items = max(iid_max + 1, 1000)
    elif dataset == 'meituan':
       n_users = max(uid_max + 1, 2145)
       n_items = max(iid_max + 1, 7189)
    elif dataset == 'yelp':
        n_users = max(uid_max + 1, 7975)
        n_items = max(iid_max + 1, 74722)
    else:
        n_users = max(uid_max + 1, 25768)
        n_items = max(iid_max + 1, 3044)
    print(f'n_users: {n_users}')  # 打印用户数量
    print(f'n_items: {n_items}')  # 打印物品数量

    valid_list = np.array(valid_list)  # 转换验证列表为NumPy数组
    test_list = np.array(test_list)  # 转换测试列表为NumPy数组

    # 构建训练数据的稀疏矩阵
    train_data = sp.csr_matrix((np.ones_like(train_list[:, 0]),
                 (train_list[:, 0], train_list[:, 1])), dtype='float64',
                 shape=(n_users, n_items))

    # 构建验证特征稀疏矩阵
    valid_x_data = sp.csr_matrix((np.ones_like(train_list[:, 0]),
                 (train_list[:, 0], train_list[:, 1])), dtype='float64',
                 shape=(n_users, n_items))
    # 构建验证标签稀疏矩阵
    valid_y_data = sp.csr_matrix((np.ones_like(valid_list[:, 0]),
                 (valid_list[:, 0], valid_list[:, 1])), dtype='float64',
                 shape=(n_users, n_items))

    # 构建测试特征稀疏矩阵
    test_x_list = train_list  # 目前的测试特征使用训练数据
    # test_x_list = np.concatenate([train_list, valid_list], 0)  # 可以选择将训练和验证数据结合作为测试数据
    test_x_data = sp.csr_matrix((np.ones_like(test_x_list[:, 0]),
                 (test_x_list[:, 0], test_x_list[:, 1])), dtype='float64',
                 shape=(n_users, n_items))
    # 构建测试标签稀疏矩阵
    test_y_data = sp.csr_matrix((np.ones_like(test_list[:, 0]),
                 (test_list[:, 0], test_list[:, 1])), dtype='float64',
                 shape=(n_users, n_items))

    return train_data, valid_x_data, valid_y_data, test_x_data, test_y_data, n_users, n_items  # 返回构建的数据和用户、物品数量


