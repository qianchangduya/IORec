import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import ipdb

# from tensorboardX import SummaryWriter
# 导入数据处理和评估相关库
from scipy import sparse
import models
import random
import data_utils
import evaluate_util
import os

# 定义超参数和配置的参数解析器
parser = argparse.ArgumentParser(description='PyTorch COR')
parser.add_argument('--model_name', type=str, default='COR', help='model name')
parser.add_argument('--dataset', type=str, default='synthetic', help='dataset name')
# parser.add_argument('--dataset', type=str, default='meituan', help='dataset name')
# parser.add_argument('--dataset', type=str, default='yelp', help='dataset name')

parser.add_argument('--data_path', type=str, default='../data/', help='directory of all datasets')
parser.add_argument('--log_name', type=str, default='', help='log/model special name')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--wd', type=float, default=0.00, help='weight decay coefficient')  # 权重衰减系数
parser.add_argument('--batch_size', type=int, default=500, help='batch size')
parser.add_argument("--mlp_dims", default='[100, 20]', help="the dims of the mlp encoder")  # MLP编码器维度
parser.add_argument("--mlp_p1_1_dims", default='[100, 200]', help="the dims of the mlp p1-1")  # MLP p1-1层维度
parser.add_argument("--mlp_p1_2_dims", default='[1]', help="the dims of the mlp p1-2")  # MLP p1-2层维度
parser.add_argument("--mlp_p2_dims", default='[]', help="the dims of the mlp p2")  # MLP p2层维度
parser.add_argument("--mlp_p3_dims", default='[10]', help="the dims of the mlp p3")  # MLP p3层维度
parser.add_argument("--Z1_hidden_size", type=int, default=8, help="hidden size of Z1")  # Z1的隐藏层大小
parser.add_argument('--E2_hidden_size', type=int, default=20, help='hidden size of E2')  # E1的隐藏层大小
parser.add_argument('--Z2_hidden_size', type=int, default=20, help='hidden size of Z2')
parser.add_argument('--total_anneal_steps', type=int, default=200000, help='the total number of gradient updates for annealing')  # 退火参数的总梯度更新步数
parser.add_argument('--anneal_cap', type=float, default=0.2, help='largest annealing parameter')  # 最大退火参数
parser.add_argument('--sample_freq', type=int, default=1, help='sample frequency for Z1/Z2')  # Z1/Z2的采样频率
parser.add_argument('--CI', type=int, default=1, help='whether use counterfactual inference in ood settings')  # 在OOD环境下启用反事实推理
parser.add_argument('--bn', type=int, default=1, help='batch norm')  # 批量归一化标志
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
parser.add_argument('--regs', type=float, default=0, help='regs')  # 正则化系数
parser.add_argument('--epochs', type=int, default=200, help='upper epoch limit')
parser.add_argument("--topN", default='[10, 20, 50, 100]', help="the recommended item num")
parser.add_argument('--cuda', action='store_true', help='use CUDA')
# parser.add_argument('--gpu', type=str, default='1', help='GPU id')
parser.add_argument('--gpu', type=str, default='0', help='GPU id')
parser.add_argument("--ood_test", default=True, help="whether test ood data during iid training")  # 在IID训练期间测试OOD数据
parser.add_argument('--save_path', type=str, default='./models/', help='path to save the final model')
parser.add_argument('--act_function', type=str, default='tanh', help='activation function')  # 激活函数
parser.add_argument('--ood_finetune',action='store_true', help='fine-tuning on ood data')  # 在OOD数据上进行微调
parser.add_argument('--ckpt', type=str, default=None, help='pre-trained model directory')  # 预训练模型目录
parser.add_argument('--X',type=int,default=10, help='use X percent of ood data for fine-tuning')  # 用于微调的OOD数据百分比
args = parser.parse_args()
print(args)

# 设置随机种子，保证实验可复现
random_seed = 1
torch.manual_seed(random_seed) # cpu
torch.cuda.manual_seed(random_seed) #gpu
np.random.seed(random_seed) #numpy
random.seed(random_seed) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn

# 定义数据加载的初始化函数
def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# device = torch.device("cuda:0" if args.cuda else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'using device {device}')

###############################################################################
# 加载数据
###############################################################################

# 设置数据路径
data_path = args.data_path + args.dataset + '/'

# 根据微调标志判断数据集类型（IID或OOD）
if args.ood_finetune:
    train_dataset = 'ood'  # 训练数据集为OOD
    ood_test_dataset = 'iid'  # OOD测试数据集为IID
else:
    train_dataset = 'iid'  # 训练数据集为IID
    ood_test_dataset = 'ood'  # OOD测试数据集为OOD

# 定义训练、验证和测试数据集路径
train_path = data_path + '{}/training_list.npy'.format(train_dataset)  # 训练集路径
valid_path = data_path + '{}/validation_dict.npy'.format(train_dataset)  # 验证集路径
test_path = data_path + '{}/testing_dict.npy'.format(train_dataset)  # 测试集路径

# 如果是OOD微调
if args.ood_finetune:
    train_path = data_path + '{}/training_list_{}%.npy'.format('X_ood', args.X)  # OOD 微调的训练集路径
    if args.X == 0:  # 如果 X 为0
        train_path = data_path + '{}/training_list.npy'.format(ood_test_dataset)  # 使用OOD测试集的训练集路径
# 根据数据集类型选择用户特征路径
if args.dataset == 'synthetic':
    user_feat_path = data_path + '{}/user_preference.npy'.format(train_dataset)  # 合成数据集的用户特征路径
else:
    user_feat_path = data_path + '{}/user_feature.npy'.format(train_dataset)  # 其他数据集的用户特征路径
item_feat_path = data_path + '{}/item_feature.npy'.format(train_dataset)  # 物品特征路径

# 加载训练和测试数据集
train_data, valid_x_data, valid_y_data, test_x_data, test_y_data, n_users, n_items = \
                                            data_utils.data_load(train_path, valid_path, test_path, args.dataset)
# 加载用户和物品特征
user_feature, item_feature = data_utils.feature_load(user_feat_path, item_feat_path)  # 加载用户和物品特征
if args.dataset == 'synthetic':
    user_feature = user_feature[:, 0:1]  # 如果是合成数据集，选择用户特征的第一列
    item_feature = torch.FloatTensor(item_feature[:, 1:9]).to(device)  # 转换物品特征为张量
else:
    item_feature = torch.FloatTensor(item_feature).to(device)  # 转换物品特征为张量

# 针对OOD数据进行测试
if args.ood_test:
    ood_train_path = data_path + '{}/training_list.npy'.format(ood_test_dataset)  # OOD训练集路径
    ood_valid_path = data_path + '{}/validation_dict.npy'.format(ood_test_dataset)  # OOD验证集路径
    ood_test_path = data_path + '{}/testing_dict.npy'.format(ood_test_dataset)  # OOD测试集路径
    if args.dataset == 'synthetic':
        ood_user_feat_path = data_path + '{}/user_preference.npy'.format(ood_test_dataset)  # 合成数据集的OOD用户特征路径
    else:
        ood_user_feat_path = data_path + '{}/user_feature.npy'.format(ood_test_dataset)  # 其他数据集的OOD用户特征路径
    ood_item_feat_path = data_path + '{}/item_feature.npy'.format(ood_test_dataset)  # OOD物品特征路径
    ood_train_data, ood_valid_x_data, ood_valid_y_data, ood_test_x_data, ood_test_y_data, ood_n_users, ood_n_items = \
        data_utils.data_load(ood_train_path, ood_valid_path, ood_test_path, args.dataset)  # 加载OOD数据集
    ood_user_feature, ood_item_feature = data_utils.feature_load(ood_user_feat_path, ood_item_feat_path)  # 加载OOD用户和物品特征
    if args.dataset == 'synthetic':
        ood_user_feature = ood_user_feature[:, 0:1]  # 如果是合成数据集，选择OOD用户特征的第一列

N = train_data.shape[0]  # 训练数据的总样本数
idxlist = list(range(N))  # 生成索引列表

# 验证和测试时的交互物品掩码
mask_val = train_data  # 验证集的交互物品掩码
mask_test = train_data + valid_y_data  # 测试集的交互物品掩码
if args.ood_finetune:
    mask_val = train_data + ood_train_data + ood_valid_y_data  # OOD微调时的验证掩码
    mask_test = train_data + valid_y_data + ood_train_data + ood_valid_y_data  # OOD微调时的测试掩码

###############################################################################
# 构建模型
###############################################################################
if args.ood_finetune:
    model = torch.load(args.ckpt)  # 加载OOD微调的模型
    ckpt_structure = args.ckpt.split('_')  # 分割检查点文件名以提取结构信息
else: 
    E1_size = user_feature.shape[1]  # 获取用户特征的维度
    Z1_size = args.Z1_hidden_size  # 获取隐藏层Z1的大小
    mlp_q_dims = [n_items + user_feature.shape[1]] + eval(args.mlp_dims) + [args.E2_hidden_size]  # Q网络的维度

    # 用于COR的结构
    mlp_p1_dims = [E1_size + args.E2_hidden_size] + eval(args.mlp_p1_1_dims) + [Z1_size]

    # 用于COR_G的结构
    mlp_p1_1_dims = [1] + eval(args.mlp_p1_1_dims)
    mlp_p1_2_dims = [mlp_p1_1_dims[-1]] + eval(args.mlp_p1_2_dims)

    mlp_p2_dims = [args.E2_hidden_size] + eval(args.mlp_p2_dims) + [args.Z2_hidden_size]
    mlp_p3_dims = [Z1_size + args.Z2_hidden_size] + eval(args.mlp_p3_dims) +  [n_items] # need to delete

    # 定义预定义的因果图
    adj = np.concatenate((np.array([[0.0]*E1_size + [1.0]*args.E2_hidden_size,
                                    [0.0]*E1_size + [1.0]*args.E2_hidden_size]),
                        np.ones([6, E1_size + args.E2_hidden_size])), axis=0)  # 构建邻接矩阵
    adj = torch.FloatTensor(adj).to(device)  # 转换为张量并移动到指定设备

    # 根据模型名称构建不同类型的模型
    if args.model_name == 'COR':
        model = models.COR(mlp_q_dims, mlp_p1_dims, mlp_p2_dims, mlp_p3_dims, \
                                                item_feature, adj, E1_size, args.dropout, args.bn, args.sample_freq, args.regs, args.act_function).to(device)
    elif args.model_name == 'COR_G':
        model = models.COR_G(mlp_q_dims, mlp_p1_1_dims, mlp_p1_2_dims, mlp_p2_dims, mlp_p3_dims, \
                                                item_feature, adj, E1_size, args.dropout, args.bn, args.sample_freq, args.regs, args.act_function).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
criterion = models.loss_function

###############################################################################
# 训练代码
###############################################################################

def naive_sparse2tensor(data):
    # 将稀疏矩阵转换为张量
    return torch.FloatTensor(data.toarray())

def adjust_lr(e):
    # 根据数据集类型调整学习率
    if args.dataset=='meituan':
        if e>90:  # 如果训练轮数超过90，降低学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.1 * args.lr
    elif args.dataset=='yelp':
        if e>60:  # 如果训练轮数超过60，降低学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.1 * args.lr
    else:
        pass  # 对于其他数据集，不调整学习率

def train():
    # 设置模型为训练模式
    model.train()
    global update_count
    np.random.shuffle(idxlist)  # 随机打乱索引列表
    
    for batch_idx, start_idx in enumerate(range(0, N, args.batch_size)):
        end_idx = min(start_idx + args.batch_size, N)  # 计算当前批次的结束索引
        data = train_data[idxlist[start_idx:end_idx]]  # 获取当前批次的训练数据
        user_f = torch.FloatTensor(user_feature[idxlist[start_idx:end_idx]]).to(device)  # 获取用户特征并转换为张量
        data = naive_sparse2tensor(data).to(device)  # 将稀疏数据转换为张量
        Z2_reuse_batch = None
        if args.ood_finetune:
            Z2_reuse_batch = Z2_reuse[idxlist[start_idx:end_idx]]  # 如果是OOD微调，获取Z2重用批次

        # 计算退火因子
        if args.total_anneal_steps > 0:
            anneal = min(args.anneal_cap, 1. * update_count / args.total_anneal_steps)
        else:
            anneal = args.anneal_cap

        optimizer.zero_grad()  # 清零梯度

        # 前向传播，获取重构数据、均值、对数方差和正则化损失
        recon_batch, mu, logvar, reg_loss = model(data, user_f, Z2_reuse_batch)

        # 计算损失
        loss = criterion(recon_batch, data, mu, logvar, anneal)
        loss = loss + reg_loss  # 加上正则化损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        update_count += 1  # 更新计数器


def evaluate(data_tr, data_te, his_mask, user_feat, topN, CI=0):
    # 评估模型性能
    assert data_tr.shape[0] == data_te.shape[0] == user_feat.shape[0]  # 确保训练数据、测试数据和用户特征的样本数量一致

    # 设置模型为评估模式
    model.eval()
    total_loss = 0.0
    global update_count
    e_idxlist = list(range(data_tr.shape[0]))  # 创建评估索引列表
    e_N = data_tr.shape[0]  # 获取评估数据的样本数量

    predict_items = []  # 预测物品列表
    target_items = []  # 目标物品列表
    for i in range(e_N):
        target_items.append(data_te[i,:].nonzero()[1].tolist())  # 获取目标物品

    with torch.no_grad():  # 在不计算梯度的情况下进行评估
        for start_idx in range(0, e_N, args.batch_size):  # 按批次处理评估数据
            end_idx = min(start_idx + args.batch_size, N)  # 计算当前批次的结束索引
            data = data_tr[e_idxlist[start_idx:end_idx]]  # 获取当前批次的训练数据
            user_f = torch.FloatTensor(user_feat[e_idxlist[start_idx:end_idx]]).to(device)  # 获取用户特征并转换为张量
            data_tensor = naive_sparse2tensor(data).to(device)  # 将稀疏数据转换为张量
            his_data = his_mask[e_idxlist[start_idx:end_idx]]  # 获取历史数据掩码
            Z2_reuse_batch = None  # 初始化Z2重用批次为None
            if args.ood_finetune and args.X != 0:  # 如果进行OOD微调且X不为0
                Z2_reuse_batch = Z2_reuse[e_idxlist[start_idx:end_idx]]  # 获取Z2重用批次

            # 计算退火因子
            if args.total_anneal_steps > 0:
                anneal = min(args.anneal_cap, 1. * update_count / args.total_anneal_steps)  # 计算退火值
            else:
                anneal = args.anneal_cap  # 如果没有退火步骤，使用最大退火值

            # 前向传播，获取重构数据、均值、对数方差和正则化损失
            recon_batch, mu, logvar, reg_loss = model(data_tensor, user_f, Z2_reuse_batch, CI)

            # 计算损失
            loss = criterion(recon_batch, data_tensor, mu, logvar, anneal)  # 计算重构损失
            loss = loss + reg_loss  # 加上正则化损失
            total_loss += loss.item()  # 累加总损失

            # 从重构数据中排除训练集中的示例
            recon_batch[his_data.nonzero()] = -np.inf  # 将历史数据的重构值设为负无穷，确保不被选择

            # 获取重构数据中前topN个物品的索引
            _, indices = torch.topk(recon_batch, topN[-1])
            indices = indices.cpu().numpy().tolist()  # 转换为列表
            predict_items.extend(indices)  # 将预测结果添加到列表中

    # 计算平均损失
    total_loss /= len(range(0, e_N, args.batch_size))
    # 计算TopN准确率
    test_results = evaluate_util.computeTopNAccuracy(target_items, predict_items, topN)
    return total_loss, test_results  # 返回总损失和测试结果

if args.ood_finetune:  # 如果进行OOD微调
    with torch.no_grad():  # 在不计算梯度的情况下进行评估
        # 获取OOD训练数据的Z2重用
        Z2_reuse = model.reuse_Z2(naive_sparse2tensor(ood_train_data).to(device),\
                            torch.FloatTensor(user_feature).to(device))  # OOD用户特征
    if args.X == 0:  # 如果X为0
        print(f"Performance on {ood_test_dataset}")  # 打印OOD测试数据集的性能
        _, test_results = evaluate(ood_test_x_data, ood_test_y_data, ood_test_x_data + ood_valid_y_data, ood_user_feature, eval(args.topN))
        evaluate_util.print_results(None, None, test_results)  # 打印测试结果

        print(f"Performance on {train_dataset}")  # 打印训练数据集的性能
        _, test_results = evaluate(ood_test_x_data, test_y_data, ood_test_x_data + ood_valid_y_data, user_feature, eval(args.topN), 1)
        evaluate_util.print_results(None, None, test_results)  # 打印测试结果
        print('-' * 18)
        print("Exiting from training by using 0 % of OOD data")  # 退出训练
        os._exit(0)  # 退出程序

# 初始化最佳召回率和相关变量
best_recall = -np.inf
best_ood_recall = -np.inf
best_epoch = 0
best_ood_epoch = 0
best_valid_results = None
best_test_results = None
best_ood_test_results = None
update_count = 0

# 根据数据集选择K值，K=0时选择recall@10，K=2时选择recall@50
K = 0 if args.dataset == 'synthetic' else 2
evaluate_interval = 1 if args.dataset=='yelp' or args.ood_finetune else 5  # 评估间隔

# 在任何时候，您都可以按Ctrl + C提前中断训练。
try:
    for epoch in range(1, args.epochs + 1):  # 进行多个训练轮次
        epoch_start_time = time.time()  # 记录当前轮次开始时间
        adjust_lr(epoch)  # 调整学习率
        train()  # 训练模型

        if epoch % evaluate_interval == 0:  # 每隔一定轮次进行评估

            # 评估验证集和测试集
            valid_loss, valid_results = evaluate(valid_x_data, valid_y_data, mask_val, user_feature, eval(args.topN))
            test_loss, test_results = evaluate(test_x_data, test_y_data, mask_test, user_feature, eval(args.topN))

            print('---'*18)
            print("Runing Epoch {:03d} ".format(epoch) + 'valid loss {:.4f}'.format(valid_loss) + " costs " + time.strftime(
                                "%H: %M: %S", time.gmtime(time.time() - epoch_start_time)))  # 打印当前轮次的损失和耗时
            evaluate_util.print_results(None, valid_results, test_results)  # 打印验证集和测试集的结果
            print('---'*18)

            if args.ood_test:  # 如果进行OOD测试
                # 评估OOD验证集和测试集
                ood_val_loss, ood_valid_results = evaluate(valid_x_data, ood_valid_y_data, mask_test,\
                                                           ood_user_feature, eval(args.topN), args.CI)
                ood_test_loss, ood_test_results = evaluate(test_x_data, ood_test_y_data, mask_test, \
                                                           ood_user_feature, eval(args.topN), args.CI)
                print(f'{ood_test_dataset} testing')  # 打印当前测试的数据集名称
                if args.ood_finetune:  # 如果进行OOD微调
                    ood_valid_results = None  # 将OOD验证结果设为None
                evaluate_util.print_results(None, ood_valid_results, ood_test_results)  # 打印OOD验证和测试结果
                print('---' * 18)  # 打印分隔线

            # 如果当前验证集的召回率是迄今为止的最佳值，则保存模型
            if valid_results[1][K] > best_recall:  # 根据召回率选择最佳模型
                best_recall, best_epoch = valid_results[1][K], epoch  # 更新最佳召回率和对应的轮次
                best_test_results = test_results  # 更新最佳测试结果
                best_valid_results = valid_results  # 更新最佳验证结果
                if args.ood_test:  # 如果进行OOD测试
                    best_ood_test_results = ood_test_results  # 更新最佳OOD测试结果
                if not os.path.exists(args.save_path):  # 如果保存路径不存在
                    os.mkdir(args.save_path)  # 创建保存路径
                if not args.ckpt is None:  # 如果指定了检查点，保存微调模型
                    torch.save(model, '{}{}_{}_{}_{}_{}_{}_{}_{}_{}_{}lr_{}wd_{}bs_{}anneal_{}cap_{}CI_{}drop_{}_{}_{}_{}bn_{}freq_{}reg_{}_{}.pth'.format(
                            args.save_path, args.model_name, args.dataset, train_dataset ,ckpt_structure[-20], ckpt_structure[-19], ckpt_structure[-18], \
                            ckpt_structure[-17], ckpt_structure[-16], ckpt_structure[-15], args.lr, args.wd,\
                            args.batch_size, args.total_anneal_steps, args.anneal_cap, args.CI, \
                            args.dropout, ckpt_structure[-7], ckpt_structure[-6], ckpt_structure[-5], args.bn, args.sample_freq, args.regs, ckpt_structure[-2], args.log_name))
                else:  # 否则保存完整模型
                    torch.save(model, '{}{}_{}_{}_{}q_{}p11_{}p12_{}p2_{}p3_{}lr_{}wd_{}bs_{}anneal_{}cap_{}CI_{}drop_{}Z1_{}E2_{}Z2_{}bn_{}freq_{}reg_{}_{}.pth'.format(
                            args.save_path, args.model_name, args.dataset, train_dataset , args.mlp_dims, args.mlp_p1_1_dims, \
                            args.mlp_p1_2_dims, args.mlp_p2_dims, args.mlp_p3_dims, args.lr, args.wd,\
                            args.batch_size, args.total_anneal_steps, args.anneal_cap, args.CI, \
                            args.dropout, args.Z1_hidden_size, args.E2_hidden_size, args.Z2_hidden_size, args.bn, args.sample_freq, args.regs, args.act_function, args.log_name))
except KeyboardInterrupt:  # 捕获键盘中断异常
    print('-' * 18)  # 打印分隔线
    print('Exiting from training early')  # 提示提前退出训练

print('===' * 18)  # 打印结束分隔线
print("End. Best Epoch {:03d} ".format(best_epoch))  # 打印最佳轮次
evaluate_util.print_results(None, best_valid_results, best_test_results)  # 打印最佳验证和测试结果
print('===' * 18)  # 打印结束分隔线
if args.ood_test:  # 如果进行OOD测试
    print(f"End. {ood_test_dataset} Performance")  # 打印OOD测试数据集的性能
    evaluate_util.print_results(None, None, best_ood_test_results)  # 打印最佳OOD测试结果
    print('===' * 18)  # 打印结束分隔线



