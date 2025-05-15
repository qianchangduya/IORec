import numpy as np
print("321")
# 加载 .npy 文件
# item_feature = np.load('item_feature.npy')
# item_feature_file = np.load('item_feature_file.npy', allow_pickle=True)
# item_mapping = np.load('item_mapping.npy', allow_pickle=True)
# item_mapping_reverse = np.load('item_mapping_reverse.npy', allow_pickle=True)
# testing_dict = np.load('testing_dict.npy', allow_pickle=True)
training_list = np.load('iid_workday_train.npy', allow_pickle=True)
user_feature = np.load('iid_user_feature.npy', allow_pickle=True)
# user_feature_file = np.load('user_feature_file.npy', allow_pickle=True)
# user_mapping = np.load('user_mapping.npy', allow_pickle=True)
validation_dict = np.load('iid_workday_val.npy', allow_pickle=True)
print("123")
# 打印数据
# print(item_feature)