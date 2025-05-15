import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# 读取 user.txt 文件
user_df = pd.read_csv('users.txt', sep='\t', header=0)

# 读取 mapping.txt 文件
mapping_df = pd.read_csv('user_id_mapping_005.txt', sep='\t', header=0)

# 筛选出 mapping.txt 中出现过的 original_user_id
user_ids_to_filter = mapping_df['original_user_id'].tolist()

# 筛选 user.txt 中的 user_id
filtered_user_df = user_df[user_df['user_id'].isin(user_ids_to_filter)]

# 创建 iid 类：只保留 user_id, avg_pay_amt, avg_pay_amt_weekdays
iid_df = filtered_user_df[['user_id', 'avg_pay_amt', 'avg_pay_amt_weekdays']]

# 创建 ood 类：只保留 user_id, avg_pay_amt, avg_pay_amt_weekends
ood_df = filtered_user_df[['user_id', 'avg_pay_amt', 'avg_pay_amt_weekends']]

# 将 mapping.txt 中的映射关系转换为字典
user_id_mapping = dict(zip(mapping_df['original_user_id'], mapping_df['new_user_id']))

# 替换 iid 和 ood 中的 user_id 为 new_user_id
iid_df['user_id'] = iid_df['user_id'].map(user_id_mapping)
ood_df['user_id'] = ood_df['user_id'].map(user_id_mapping)

# 对数据按 user_id 列排序，保持原有行不变
iid_df_sorted = iid_df.sort_values(by='user_id', ascending=True)
ood_df_sorted = ood_df.sort_values(by='user_id', ascending=True)

# 保存排序后的数据到新文件
iid_df_sorted.to_csv('iid_users.txt', sep='\t', index=False)
ood_df_sorted.to_csv('ood_users.txt', sep='\t', index=False)

print("数据已分为 'iid_users.txt' 和 'ood_users.txt' 文件。")


# 定义区间映射函数
def map_interval(value):
    mapping = {
        '<29': 14.5,
        '[29,36)': 32.5,
        '[36,49)': 42.5,
        '[49,65)': 57,
        '>=65': 70  # 将 >=65 映射为 70
    }
    return mapping.get(value, np.nan)  # 对于未知值返回 NaN

# 读取 iid_user_sorted.txt 和 ood_user_sorted.txt 文件
iid_DF = pd.read_csv('iid_users.txt', sep='\t')
ood_DF = pd.read_csv('ood_users.txt', sep='\t')

# 转换区间数据为数值
iid_DF['avg_pay_amt'] = iid_DF['avg_pay_amt'].apply(map_interval)
iid_DF['avg_pay_amt_weekdays'] = iid_DF['avg_pay_amt_weekdays'].apply(map_interval)

ood_DF['avg_pay_amt'] = ood_DF['avg_pay_amt'].apply(map_interval)
ood_DF['avg_pay_amt_weekends'] = ood_DF['avg_pay_amt_weekends'].apply(map_interval)

# 初始化 KNNImputer
imputer = KNNImputer(n_neighbors=5)

# 对含有缺失值的特征进行插补
iid_features = iid_DF[['avg_pay_amt', 'avg_pay_amt_weekdays']].values
ood_features = ood_DF[['avg_pay_amt', 'avg_pay_amt_weekends']].values

iid_features_imputed = imputer.fit_transform(iid_features)
ood_features_imputed = imputer.transform(ood_features)

# 将插补后的特征保存为新的 DataFrame
iid_user_feature = pd.DataFrame(iid_features_imputed, columns=['avg_pay_amt', 'avg_pay_amt_weekdays'])
ood_user_feature = pd.DataFrame(ood_features_imputed, columns=['avg_pay_amt', 'avg_pay_amt_weekends'])

# 保存处理后的特征文件
iid_user_feature.to_csv('iid_user_feature.csv', sep='\t', index=False)
ood_user_feature.to_csv('ood_user_feature.csv', sep='\t', index=False)

print("特征文件已保存为 'iid_user_feature.csv' 和 'ood_user_feature.csv'。")