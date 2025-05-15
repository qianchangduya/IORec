import argparse  # 导入参数解析库
import time  # 导入时间库
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块
import torch.optim as optim  # 导入PyTorch优化器模块
import numpy as np  # 导入NumPy库

# from tensorboardX import SummaryWriter  # 导入TensorBoardX（可选）
from scipy import sparse  # 导入SciPy稀疏矩阵模块
import models  # 导入模型模块
import random  # 导入随机数生成库
import data_utils  # 导入数据处理工具模块
import evaluate_util  # 导入评估工具模块
import os  # 导入操作系统接口模块

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='PyTorch COR Inference')
parser.add_argument('--model_name', type=str, default='COR',
                    help='model name')  # 模型名称
# parser.add_argument('--dataset', type=str, default='synthetic', help='dataset name')  # 数据集名称
parser.add_argument('--dataset', type=str, default='meituan', help='dataset name')  # 数据集名称
parser.add_argument('--data_path', type=str, default='../data/',
                    help='directory of all datasets')  # 数据集目录
parser.add_argument('--batch_size', type=int, default=500,
                    help='batch size')  # 批处理大小
parser.add_argument('--CI', type=int, default=1,
                    help='whether use counterfactual inference in ood settings')  # 是否在OOD设置中使用反事实推断
parser.add_argument("--topN", default='[10, 20, 50, 100]',
                    help="the recommended item num")  # 推荐的物品数量
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')  # 是否使用CUDA
parser.add_argument('--gpu', type=str, default='0',
                    help='GPU id')  # GPU ID
# parser.add_argument('--ckpt', type=str, default=None,
#                     help='pre-trained best iid model path')  # 预训练的最佳IID模型路径
# parser.add_argument('--ckpt', type=str, default= r'E:\IPG-Rec-master\COR-main\code\models\COR_synthetic_iid_[100, 20]q_[100, 200]p11_[1]p12_[]p2_[10]p3_0.0001lr_0.0wd_500bs_200000anneal_0.2cap_1CI_0.5drop_8Z1_20E2_20Z2_1bn_1freq_0reg_tanh_.pth', help='pre-trained best iid model path')
parser.add_argument('--ckpt', type=str, default= r'E:\IPG-Rec-master\COR-main\code\models\COR_meituan_iid_[3000]q_[]p11_[1]p12_[]p2_[]p3_0.001lr_0.0wd_500bs_0anneal_0.1cap_1CI_0.5drop_500Z1_1000E2_200Z2_0bn_1freq_0.0reg_tanh_log.pth', help='pre-trained best iid model path')


# 解析命令行参数
args = parser.parse_args()
print(args)  # 打印解析的参数

# 设置随机种子以确保实验的可重复性
random_seed = 1
torch.manual_seed(random_seed)  # 设置CPU随机种子
torch.cuda.manual_seed(random_seed)  # 设置GPU随机种子
np.random.seed(random_seed)  # 设置NumPy随机种子
random.seed(random_seed)  # 设置Python内置随机数生成器的种子
torch.backends.cudnn.deterministic = True  # 设置cuDNN为确定性模式

# 初始化工作线程的随机种子
def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)  # 每个工作线程使用不同的随机种子

# 设置可见的CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# 选择使用的设备（CUDA或CPU）
# device = torch.device("cuda:0" if args.cuda else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'using device {device}')  # 打印使用的设备

###############################################################################
# 加载数据
###############################################################################
data_path = args.data_path + args.dataset + '/'  # 构建数据路径

# 加载IID数据
iid_test_dataset = 'iid'  # IID测试数据集标识
train_path = data_path + '{}/training_list.npy'.format(iid_test_dataset)  # 训练数据路径
valid_path = data_path + '{}/validation_dict.npy'.format(iid_test_dataset)  # 验证数据路径
test_path = data_path + '{}/testing_dict.npy'.format(iid_test_dataset)  # 测试数据路径

# 根据数据集类型加载用户特征路径
if args.dataset == 'synthetic':
    user_feat_path = data_path + '{}/user_preference.npy'.format(iid_test_dataset)  # 合成数据集用户偏好路径
else:
    user_feat_path = data_path + '{}/user_feature.npy'.format(iid_test_dataset)  # 其他数据集用户特征路径

item_feat_path = data_path + '{}/item_feature.npy'.format(iid_test_dataset)  # 物品特征路径

# 加载训练、验证和测试数据
train_data, valid_x_data, valid_y_data, test_x_data, test_y_data, n_users, n_items = \
    data_utils.data_load(train_path, valid_path, test_path, args.dataset)  # 加载数据集
user_feature, item_feature = data_utils.feature_load(user_feat_path, item_feat_path)  # 加载特征

# 处理合成数据集的用户和物品特征
if args.dataset == 'synthetic':
    user_feature = user_feature[:, 0:1]  # 仅保留用户特征的第一列
    item_feature = torch.FloatTensor(item_feature[:, 1:9]).to(device)  # 转换物品特征为张量并移动到设备
else:
    item_feature = torch.FloatTensor(item_feature).to(device)  # 转换物品特征为张量并移动到设备

# 加载OOD数据
ood_test_dataset = 'ood'  # OOD测试数据集标识
ood_train_path = data_path + '{}/training_list.npy'.format(ood_test_dataset)  # OOD训练数据路径
ood_valid_path = data_path + '{}/validation_dict.npy'.format(ood_test_dataset)  # OOD验证数据路径
ood_test_path = data_path + '{}/testing_dict.npy'.format(ood_test_dataset)  # OOD测试数据路径

# 根据数据集类型加载OOD用户特征路径
if args.dataset == 'synthetic':
    ood_user_feat_path = data_path + '{}/user_preference.npy'.format(ood_test_dataset)  # 合成数据集OOD用户偏好路径
else:
    ood_user_feat_path = data_path + '{}/user_feature.npy'.format(ood_test_dataset)  # 其他数据集OOD用户特征路径

ood_item_feat_path = data_path + '{}/item_feature.npy'.format(ood_test_dataset)  # OOD物品特征路径

# 加载OOD训练、验证和测试数据
ood_train_data, ood_valid_x_data, ood_valid_y_data, ood_test_x_data, ood_test_y_data, ood_n_users, ood_n_items = \
    data_utils.data_load(ood_train_path, ood_valid_path, ood_test_path, args.dataset)  # 加载OOD数据集
ood_user_feature, ood_item_feature = data_utils.feature_load(ood_user_feat_path, ood_item_feat_path)  # 加载OOD特征

# 处理合成数据集的OOD用户特征
if args.dataset == 'synthetic':
    ood_user_feature = ood_user_feature[:, 0:1]  # 仅保留OOD用户特征的第一列

# 将训练数据和验证标签合并为一个掩码
mask_test =  train_data + valid_y_data

N = train_data.shape[0]  # 获取训练数据的样本数
idxlist = list(range(N))  # 创建索引列表

###############################################################################
# 加载模型
###############################################################################

# 从指定路径加载模型
model = torch.load(args.ckpt)

# 定义损失函数
criterion = models.loss_function

###############################################################################
# 推断
###############################################################################

def naive_sparse2tensor(data):
    # 定义将稀疏矩阵转换为张量的函数
    return torch.FloatTensor(data.toarray())  # 转换为FloatTensor类型

# 定义评估函数
def evaluate(data_tr, data_te, his_mask, user_feat, topN, CI=0):
    # 确保训练数据、测试数据和用户特征的样本数相同
    assert data_tr.shape[0] == data_te.shape[0] == user_feat.shape[0]

    # 切换到评估模式
    model.eval()
    total_loss = 0.0  # 初始化总损失
    e_idxlist = list(range(data_tr.shape[0]))  # 重新创建索引列表
    e_N = data_tr.shape[0]  # 获取当前样本数

    predict_items = []  # 用于存储预测的物品
    target_items = []  # 用于存储目标物品
    for i in range(e_N):
        target_items.append(data_te[i, :].nonzero()[1].tolist())  # 收集目标物品的索引

    with torch.no_grad():  # 禁用梯度计算
        # 按批次进行推断
        for start_idx in range(0, e_N, args.batch_size):
            end_idx = min(start_idx + args.batch_size, N)  # 计算批次结束索引
            data = data_tr[e_idxlist[start_idx:end_idx]]  # 获取当前批次的训练数据
            his_item = his_mask[e_idxlist[start_idx:end_idx]]  # 获取历史物品掩码
            user_f = torch.FloatTensor(user_feat[e_idxlist[start_idx:end_idx]]).to(device)  # 获取用户特征
            data_tensor = naive_sparse2tensor(data).to(device)  # 将稀疏矩阵转换为张量并移动到设备
            Z2_reuse_batch = None  # 用于保存模型重用的参数

            # 前向传播获取重构输出和潜变量
            recon_batch, mu, logvar, _ = model(data_tensor, user_f, Z2_reuse_batch, CI)

            # 计算损失
            loss = criterion(recon_batch, data_tensor, mu, logvar)
            total_loss += loss.item()  # 累加损失

            # 排除训练集中已有的物品
            recon_batch[his_item.nonzero()] = -np.inf  # 将历史物品的重构输出置为负无穷

            # 获取前topN个物品
            _, indices = torch.topk(recon_batch, topN[-1])  # 获取预测的索引
            indices = indices.cpu().numpy().tolist()  # 转换为numpy列表
            predict_items.extend(indices)  # 添加预测物品

    total_loss /= len(range(0, e_N, args.batch_size))  # 计算平均损失
    test_results = evaluate_util.computeTopNAccuracy(target_items, predict_items, topN)  # 计算Top-N准确率

    return total_loss, test_results  # 返回总损失和评估结果


print('Inference COR on iid and ood data')  # 打印推断信息

print('---' * 18)  # 分隔线
print('Given IID Interaction')  # 说明给定的IID交互
print("Performance on iid")  # 打印IID模式下的性能

# 在IID数据上进行验证和测试
valid_loss, valid_results = evaluate(valid_x_data, valid_y_data, valid_x_data, user_feature, eval(args.topN))
test_loss, test_results = evaluate(test_x_data, test_y_data, mask_test, user_feature, eval(args.topN))
evaluate_util.print_results(None, valid_results, test_results)  # 打印验证和测试结果

print("Performance on ood")  # 打印OOD模式下的性能
# 在OOD数据上进行验证和测试
valid_loss, valid_results = evaluate(valid_x_data, ood_valid_y_data, mask_test, ood_user_feature, eval(args.topN), args.CI)
test_loss, test_results = evaluate(test_x_data, ood_test_y_data, mask_test, ood_user_feature, eval(args.topN), args.CI)
evaluate_util.print_results(None, valid_results, test_results)  # 打印OOD验证和测试结果

print('---' * 18)  # 分隔线
