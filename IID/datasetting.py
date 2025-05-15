import numpy as np
import pandas as pd

# 1. 加载 demo6_iid_users.npy 文件
user_ids = np.load('demo6_iid_users.npy')

# 2. 获取布尔数组中 True 值的索引
true_indices = np.where(user_ids)[0]

# 3. 读取 iid_workday.txt 文件
df = pd.read_csv('iid_workday.txt', sep='\t')

# 4. 筛选出在 demo6_iid_users.npy 中标记为 True 的 user_id_new
filtered_df = df[df['user_id_new'].isin(true_indices)]

# 5. 按 'user_id_new' 列升序排序
df_sorted = filtered_df.sort_values(by='user_id_new')

# 6. 构建映射关系：将每个 user_id_new 映射到对应的 wm_poi_id_new 列表
user_to_pois = df_sorted.groupby('user_id_new')['wm_poi_id_new'].apply(list).to_dict()

# 7. 生成转换后的数据
result = []
for user_id, poi_ids in user_to_pois.items():
    result.append(f"{user_id} " + " ".join(map(str, poi_ids)))

# 8. 保存转换后的数据为 'filtered_iid.txt' 文件
with open('iid.txt', 'w') as f:
    for line in result:
        f.write(line + '\n')
