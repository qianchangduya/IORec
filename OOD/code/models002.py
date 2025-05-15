import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import ipdb


class COR(nn.Module):
    '''
    此模型不使用 item feature，预计未来会扩展使用 item feature。
    '''

    def __init__(self, mlp_q_dims, mlp_p1_dims, mlp_p2_dims, mlp_p3_dims,
                adj, E1_size, dropout=0.5, bn=0, sample_freq=1, regs=0, act_function='tanh'):
        super(COR, self).__init__()
        self.mlp_q_dims = mlp_q_dims  # q 网络的维度
        self.mlp_p1_dims = mlp_p1_dims  # p1 网络的维度
        self.mlp_p2_dims = mlp_p2_dims  # p2 网络的维度
        self.mlp_p3_dims = mlp_p3_dims  # p3 网络的维度
        self.adj = adj  # 邻接矩阵
        self.E1_size = E1_size  # E1 的大小
        self.bn = bn  # 是否使用批归一化
        self.sample_freq = sample_freq  # 采样频率
        self.regs = regs  # 正则化参数

        # 设置激活函数
        if act_function == 'tanh':
            self.act_function = F.tanh
        elif act_function == 'sigmoid':
            self.act_function = F.sigmoid

        # item_feature 在本模型中未使用，未来可能扩展使用

        # q 网络最后一维用于均值和方差
        temp_q_dims = self.mlp_q_dims[:-1] + [self.mlp_q_dims[-1] * 2]
        temp_p1_dims = self.mlp_p1_dims[:-1] + [self.mlp_p1_dims[-1] * 2]
        temp_p2_dims = self.mlp_p2_dims[:-1] + [self.mlp_p2_dims[-1] * 2]
        temp_p3_dims = self.mlp_p3_dims

        # 定义各个 MLP 层
        self.mlp_q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                           d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.mlp_p1_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                            d_in, d_out in zip(temp_p1_dims[:-1], temp_p1_dims[1:])])
        self.mlp_p2_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                            d_in, d_out in zip(temp_p2_dims[:-1], temp_p2_dims[1:])])
        self.mlp_p3_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                            d_in, d_out in zip(temp_p3_dims[:-1], temp_p3_dims[1:])])

        self.drop = nn.Dropout(dropout)  # dropout 层
        if self.bn:
            self.batchnorm = nn.BatchNorm1d(E1_size)  # 批归一化层

        self.init_weights()  # 初始化权重

    def reuse_Z2(self, D, E1):
        # 重新使用 Z2 进行推理
        if self.bn:
            E1 = self.batchnorm(E1)  # 应用批归一化
        D = F.normalize(D)  # 归一化 D 张量
        mu, _ = self.encode(torch.cat((D, E1), 1))  # 进行编码以获取均值
        h_p2 = mu
        for i, layer in enumerate(self.mlp_p2_layers):
            h_p2 = layer(h_p2)  # 通过 p2 网络的每一层
            if i != len(self.mlp_p2_layers) - 1:
                h_p2 = F.tanh(h_p2)  # 应用激活函数
            else:
                h_p2 = h_p2[:, :self.mlp_p2_dims[-1]]  # 只保留最后一层前的维度
        return h_p2

    def forward(self, D, E1, Z2_reuse=None, CI=0):
        # 向前传播
        if self.bn:
            E1 = self.batchnorm(E1)  # 应用批归一化
        D = F.normalize(D)  # 归一化 D 张量
        encoder_input = torch.cat((D, E1), 1)  # 将 D 和 E1 拼接为输入
        mu, logvar = self.encode(encoder_input)  # 编码以获取均值和对数方差
        E2 = self.reparameterize(mu, logvar)  # 通过重参数化获取 E2

        if CI == 1:  # 如果 CI 为 1，D 为 NULL
            encoder_input_0 = torch.cat((torch.zeros_like(D), E1), 1)  # 用零填充 D
            mu_0, logvar_0 = self.encode(encoder_input_0)  # 编码填充输入获取均值和方差
            E2_0 = self.reparameterize(mu_0, logvar_0)  # 重参数化获取 E2_0
            scores = self.decode(E1, E2, E2_0, Z2_reuse)  # 解码得到分数
        else:
            scores = self.decode(E1, E2, None, Z2_reuse)  # 解码得到分数
        reg_loss = self.reg_loss()  # 计算正则化损失
        return scores, mu, logvar, reg_loss  # 返回分数和损失

    def encode(self, encoder_input):
        # 编码函数
        h = self.drop(encoder_input)  # 应用 dropout
        for i, layer in enumerate(self.mlp_q_layers):
            h = layer(h)  # 向前传播通过 q 网络的层
            if i != len(self.mlp_q_layers) - 1:
                h = self.act_function(h)  # 应用激活函数
            else:
                mu = h[:, :self.mlp_q_dims[-1]]  # 分离均值
                logvar = h[:, self.mlp_q_dims[-1]:]  # 分离对数方差
        return mu, logvar  # 返回均值和对数方差

    def reparameterize(self, mu, logvar):
        # 重参数化技巧，用于从正态分布中采样
        if self.training:
            std = torch.exp(0.5 * logvar)  # 计算标准差
            eps = torch.randn_like(std)  # 生成与标准差相同形状的随机噪声
            return eps.mul(std).add_(mu)  # 返回采样后的值
        else:
            return mu  # 训练模式之外，直接返回均值

    def decode(self, E1, E2, E2_0=None, Z2_reuse=None):
        # 解码函数，将隐变量转化为可用输出
        if E2_0 is None:
            h_p1 = torch.cat((E1, E2), 1)  # 拼接 E1 和 E2 作为输入
        else:
            h_p1 = torch.cat((E1, E2_0), 1)  # 如果提供 E2_0，拼接 E1 和 E2_0

        # 处理 p1 的前向传播
        for i, layer in enumerate(self.mlp_p1_layers):  # 对 p1 网络的每一层进行前向传播
            h_p1 = layer(h_p1)  # 通过当前层
            if i != len(self.mlp_p1_layers) - 1:
                h_p1 = self.act_function(h_p1)  # 在最后一层之前应用激活函数
            else:
                Z1_mu = h_p1[:, :self.mlp_p1_dims[-1]]  # 提取均值
                Z1_logvar = h_p1[:, self.mlp_p1_dims[-1]:]  # 提取对数方差

        # 处理 p2 的前向传播
        h_p2 = E2  # 将 E2 赋值给 h_p2
        for i, layer in enumerate(self.mlp_p2_layers):  # 对 p2 网络进行前向传播
            h_p2 = layer(h_p2)  # 通过当前层
            if i != len(self.mlp_p2_layers) - 1:
                h_p2 = self.act_function(h_p2)  # 在最后一层之前应用激活函数
            else:
                Z2_mu = h_p2[:, :self.mlp_p2_dims[-1]]  # 提取均值
                Z2_logvar = h_p2[:, self.mlp_p2_dims[-1]:]  # 提取对数方差

        # 如果提供了 Z2_reuse，使用这个重用的 Z2；
        # 否则，根据采样频率生成新的 Z1 和 Z2
        if Z2_reuse is not None:
            for i in range(self.sample_freq):  # 按照采样频率进行多次采样
                if i == 0:
                    Z1 = self.reparameterize(Z1_mu, Z1_logvar)  # 第一次采样 Z1
                    Z1 = torch.unsqueeze(Z1, 0)  # 增加一个维度以便后续合并
                else:
                    Z1_ = self.reparameterize(Z1_mu, Z1_logvar)  # 后续采样 Z1
                    Z1_ = torch.unsqueeze(Z1_, 0)  # 增加维度
                    Z1 = torch.cat([Z1, Z1_], 0)  # 合并所有采样结果
            Z1 = torch.mean(Z1, 0)  # 对合并后的结果进行平均
            Z2 = Z2_reuse  # 使用提供的 Z2
        else:
            for i in range(self.sample_freq):
                if i == 0:
                    Z1 = self.reparameterize(Z1_mu, Z1_logvar)  # 第一次采样 Z1
                    Z2 = self.reparameterize(Z2_mu, Z2_logvar)  # 第一次采样 Z2
                    Z1 = torch.unsqueeze(Z1, 0)  # 增加维度
                    Z2 = torch.unsqueeze(Z2, 0)  # 增加维度
                else:
                    Z1_ = self.reparameterize(Z1_mu, Z1_logvar)  # 后续采样 Z1
                    Z2_ = self.reparameterize(Z2_mu, Z2_logvar)  # 后续采样 Z2
                    Z1_ = torch.unsqueeze(Z1_, 0)  # 增加维度
                    Z2_ = torch.unsqueeze(Z2_, 0)  # 增加维度
                    Z1 = torch.cat([Z1, Z1_], 0)  # 合并 Z1
                    Z2 = torch.cat([Z2, Z2_], 0)  # 合并 Z2
            Z1 = torch.mean(Z1, 0)  # 对 Z1 的所有采样结果进行平均
            Z2 = torch.mean(Z2, 0)  # 对 Z2 的所有采样结果进行平均

        user_preference = torch.cat((Z1, Z2), 1)  # 合并 Z1 和 Z2 作为用户偏好

        # 处理 p3 的前向传播
        h_p3 = user_preference  # 将用户偏好赋值给 h_p3
        for i, layer in enumerate(self.mlp_p3_layers):  # 对 p3 网络进行前向传播
            h_p3 = layer(h_p3)  # 通过当前层
            if i != len(self.mlp_p3_layers) - 1:
                h_p3 = self.act_function(h_p3)  # 在最后一层之前应用激活函数
        return h_p3  # 返回最终结果

    def init_weights(self):
        # 初始化权重函数
        for layer in self.mlp_q_layers:  # 对 q 网络的每一层进行初始化
            # Xavier 初始化权重
            size = layer.weight.size()  # 获取权重的形状
            fan_out = size[0]  # 输出节点数
            fan_in = size[1]  # 输入节点数
            std = np.sqrt(2.0 / (fan_in + fan_out))  # 计算标准差
            layer.weight.data.normal_(0.0, std)  # 正态分布初始化权重

            # 对偏置进行常规初始化
            layer.bias.data.normal_(0.0, 0.001)  # 将偏置初始化为接近零的值

        for layer in self.mlp_p1_layers:  # 对 p1 网络进行初始化
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.mlp_p2_layers:  # 对 p2 网络进行初始化
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.mlp_p3_layers:  # 对 p3 网络进行初始化
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

    def reg_loss(self):
        r"""计算模型参数的 L2 正则化损失，包括嵌入矩阵和模型的权重矩阵。
        返回:
            loss(torch.FloatTensor): L2 损失张量。形状为 [1,]
        """
        reg_loss = 0  # 初始化正则化损失
        # 遍历 q 网络中所有的参数
        for name, parm in self.mlp_q_layers.named_parameters():
            if name.endswith('weight'):  # 只计算权重的正则化损失
                reg_loss += self.regs * (1 / 2) * parm.norm(2).pow(2)  # L2正则化损失

        # 遍历 p1 网络中所有的参数
        for name, parm in self.mlp_p1_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss += self.regs * (1 / 2) * parm.norm(2).pow(2)  # L2正则化损失

        # 遍历 p2 网络中所有的参数
        for name, parm in self.mlp_p2_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss += self.regs * (1 / 2) * parm.norm(2).pow(2)  # L2正则化损失

        # 遍历 p3 网络中所有的参数
        for name, parm in self.mlp_p3_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss += self.regs * (1 / 2) * parm.norm(2).pow(2)  # L2正则化损失

        return reg_loss  # 返回计算得到的正则化损失


class COR_G(nn.Module):
    """
    代码中添加因果图
    模型中未使用物品特征的扩展
    """

    def __init__(self, mlp_q_dims, mlp_p1_1_dims, mlp_p1_2_dims, mlp_p2_dims, mlp_p3_dims,
                 item_feature, adj, E1_size, dropout=0.5, bn=0, sample_freq=1, regs=0, act_function='tanh'):
        super(COR_G, self).__init__()
        # 初始化网络结构的参数
        self.mlp_q_dims = mlp_q_dims
        self.mlp_p1_1_dims = mlp_p1_1_dims
        self.mlp_p1_2_dims = mlp_p1_2_dims
        self.mlp_p2_dims = mlp_p2_dims
        self.mlp_p3_dims = mlp_p3_dims
        self.adj = adj
        self.E1_size = E1_size
        self.Z1_size = adj.size(0)
        self.bn = bn
        self.sample_freq = sample_freq
        self.regs = regs

        # 选择激活函数
        if act_function == 'tanh':
            self.act_function = F.tanh
        elif act_function == 'sigmoid':
            self.act_function = F.sigmoid

        # 未在此模型中使用物品特征，留作未来使用
        self.item_feature = item_feature
        self.item_learnable_dim = self.mlp_p2_dims[-1]
        self.item_learnable_feat = torch.randn([self.item_feature.size(0), self.item_learnable_dim],
                                               requires_grad=True).cuda()

        # q 网络最后一层的维度是均值和方差
        temp_q_dims = self.mlp_q_dims[:-1] + [self.mlp_q_dims[-1] * 2]
        temp_p1_1_dims = self.mlp_p1_1_dims
        temp_p1_2_dims = self.mlp_p1_2_dims[:-1] + [self.mlp_p1_2_dims[-1] * 2]
        temp_p2_dims = self.mlp_p2_dims[:-1] + [self.mlp_p2_dims[-1] * 2]
        temp_p3_dims = self.mlp_p3_dims

        # 构建各层的线性模型
        self.mlp_q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                           d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.mlp_p1_1_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                              d_in, d_out in zip(temp_p1_1_dims[:-1], temp_p1_1_dims[1:])])

        # 构建 p1 的第二部分层，使用可训练的张量
        self.mlp_p1_2_layers = [(torch.randn([self.Z1_size, d_in, d_out], requires_grad=True)).cuda() for
                                d_in, d_out in zip(temp_p1_2_dims[:-1], temp_p1_2_dims[1:])]

        # 将张量展平并转换为参数
        for i, matrix in enumerate(self.mlp_p1_2_layers):
            temp = torch.unsqueeze(matrix, 0) if i == 0 else torch.cat((temp, torch.unsqueeze(matrix, 0)), 0)
        self.mlp_p1_2_layers = nn.Parameter(temp)

        # 构建其他层
        self.mlp_p2_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                            d_in, d_out in zip(temp_p2_dims[:-1], temp_p2_dims[1:])])
        self.mlp_p3_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                            d_in, d_out in zip(temp_p3_dims[:-1], temp_p3_dims[1:])])

        self.drop = nn.Dropout(dropout)  # dropout层
        if self.bn:
            self.batchnorm = nn.BatchNorm1d(E1_size)  # 批归一化层
        self.init_weights()  # 初始化权重

    def reuse_Z2(self, D, E1):
        if self.bn:
            E1 = self.batchnorm(E1)  # 应用批归一化
        D = F.normalize(D)  # 归一化 D
        mu, _ = self.encode(torch.cat((D, E1), 1))  # 编码获取均值
        h_p2 = mu
        for i, layer in enumerate(self.mlp_p2_layers):
            h_p2 = layer(h_p2)  # 前向传播
            if i != len(self.mlp_p2_layers) - 1:
                h_p2 = F.tanh(h_p2)  # 应用激活函数
            else:
                h_p2 = h_p2[:, :self.mlp_p2_dims[-1]]  # 取最后一层的输出
        return h_p2  # 返回 h_p2 的输出

    def forward(self, D, E1, Z2_reuse=None, CI=0):
        if self.bn:  # 如果启用批归一化
            E1 = self.batchnorm(E1)  # 对 E1 进行批归一化
        encoder_input = torch.cat((D, E1), 1)  # 将 D 和 E1 进行拼接作为编码器输入
        mu, logvar = self.encode(encoder_input)  # 对编码器输入进行编码，获得均值 mu 和对数方差 logvar
        E2 = self.reparameterize(mu, logvar)  # 使用重参数化方法获取 E2

        if CI == 1:  # 如果 CI 等于 1
            encoder_input_0 = torch.cat((torch.zeros_like(D), E1), 1)  # 创建一个 D 为零的输入
            mu_0, logvar_0 = self.encode(encoder_input_0)  # 编码新的输入
            E2_0 = self.reparameterize(mu_0, logvar_0)  # 获取 E2_0
            scores = self.decode(E1, E2, E2_0, Z2_reuse)  # 解码过程
        else:
            scores = self.decode(E1, E2, None, Z2_reuse)  # 解码且不使用 E2_0
        reg_loss = self.reg_loss()  # 计算正则化损失
        return scores, mu, logvar, reg_loss  # 返回得分、均值、方差和正则化损失

    def encode(self, encoder_input):
        h = self.drop(encoder_input)  # 应用 dropout
        for i, layer in enumerate(self.mlp_q_layers):  # 遍历 q 网络中的每一层
            h = layer(h)  # 前向传播
            if i != len(self.mlp_q_layers) - 1:  # 如果不是最后一层
                h = self.act_function(h)  # 应用激活函数
            else:
                mu = h[:, :self.mlp_q_dims[-1]]  # 获取均值
                logvar = h[:, self.mlp_q_dims[-1]:]  # 获取对数方差

        return mu, logvar  # 返回均值和对数方差

    def reparameterize(self, mu, logvar):
        if self.training:  # 如果模型在训练中
            std = torch.exp(0.5 * logvar)  # 计算标准差
            eps = torch.randn_like(std)  # 生成与标准差形状相同的随机噪声
            return eps.mul(std).add_(mu)  # 使用重参数化技巧，返回重参数化的样本
        else:
            return mu  # 在评估阶段直接返回均值

    def decode(self, E1, E2, E2_0=None, Z2_reuse=None):
        if E2_0 is None:
            h_p1 = torch.cat((E1, E2), 1)  # 如果 E2_0 为 None，拼接 E1 和 E2
        else:
            h_p1 = torch.cat((E1, E2_0), 1)  # 否则使用 E2_0

        h_p1 = torch.unsqueeze(h_p1, -1)  # 在最后一个维度增加一个维度
        for i, layer in enumerate(self.mlp_p1_1_layers):  # 遍历 p1 网络的第一部分
            h_p1 = layer(h_p1)  # 前向传播
            if i != len(self.mlp_p1_1_layers) - 1:
                h_p1 = self.act_function(h_p1)  # 应用激活函数

        h_p1 = torch.matmul(self.adj, h_p1)  # 使用邻接矩阵进行图卷积操作
        h_p1 = torch.unsqueeze(h_p1, 2)  # 增加一维以适应后续处理
        for i, matrix in enumerate(self.mlp_p1_2_layers):  # 遍历 p1 网络的第二部分
            h_p1 = torch.matmul(h_p1, matrix)  # 矩阵乘法
            if i != len(self.mlp_p1_2_layers) - 1:
                h_p1 = self.act_function(h_p1)  # 应用激活函数
            else:
                h_p1 = torch.squeeze(h_p1)  # 最后一层需去掉冗余维度
                Z1_mu = torch.squeeze(h_p1[:, :, :self.mlp_p1_2_dims[-1]])  # 获取 Z1 的均值
                Z1_logvar = torch.squeeze(h_p1[:, :, self.mlp_p1_2_dims[-1]:])  # 获取 Z1 的方差

        h_p2 = E2  # 将 E2 赋值给 h_p2
        for i, layer in enumerate(self.mlp_p2_layers):  # 遍历 p2 网络
            h_p2 = layer(h_p2)  # 前向传播
            if i != len(self.mlp_p2_layers) - 1:
                h_p2 = self.act_function(h_p2)  # 应用激活函数
            else:
                Z2_mu = h_p2[:, :self.mlp_p2_dims[-1]]  # 获取 Z2 的均值
                Z2_logvar = h_p2[:, self.mlp_p2_dims[-1]:]  # 获取 Z2 的方差

        # 生成 Z1 和 Z2
        if Z2_reuse is not None:
            for i in range(self.sample_freq):  # 遍历采样次数
                if i == 0:
                    Z1 = self.reparameterize(Z1_mu, Z1_logvar)  # 重参数化
                    Z1 = torch.unsqueeze(Z1, 0)  # 增加维度
                else:
                    Z1_ = self.reparameterize(Z1_mu, Z1_logvar)
                    Z1_ = torch.unsqueeze(Z1_, 0)  # 增加维度
                    Z1 = torch.cat([Z1, Z1_], 0)  # 合并采样结果
            Z1 = torch.mean(Z1, 0)  # 对采样结果取平均
            Z2 = Z2_reuse  # 复用 Z2
        else:
            for i in range(self.sample_freq):  # 遍历采样次数
                if i == 0:
                    Z1 = self.reparameterize(Z1_mu, Z1_logvar)  # 重参数化
                    Z2 = self.reparameterize(Z2_mu, Z2_logvar)  # 重参数化
                    Z1 = torch.unsqueeze(Z1, 0)  # 增加维度
                    Z2 = torch.unsqueeze(Z2, 0)  # 增加维度
                else:
                    Z1_ = self.reparameterize(Z1_mu, Z1_logvar)  # 重参数化
                    Z2_ = self.reparameterize(Z2_mu, Z2_logvar)  # 重参数化
                    Z1_ = torch.unsqueeze(Z1_, 0)  # 增加维度
                    Z2_ = torch.unsqueeze(Z2_, 0)  # 增加维度
                    Z1 = torch.cat([Z1, Z1_], 0)  # 合并采样结果
                    Z2 = torch.cat([Z2, Z2_], 0)  # 合并采样结果
            Z1 = torch.mean(Z1, 0)  # 对 Z1 取平均
            Z2 = torch.mean(Z2, 0)  # 对 Z2 取平均

        user_preference = torch.cat((Z1, Z2), 1)  # 合并 Z1 和 Z2 生成用户偏好

        h_p3 = user_preference  # h_p3 初始化为用户偏好
        for i, layer in enumerate(self.mlp_p3_layers):  # 遍历 p3 网络中的每一层
            h_p3 = layer(h_p3)  # 对 h_p3 进行前向传播
            if i != len(self.mlp_p3_layers) - 1:  # 如果不是最后一层
                h_p3 = self.act_function(h_p3)  # 应用激活函数
        return h_p3  # 返回最终的 h_p3 结果

    def init_weights(self):
        # 初始化模型的权重
        for layer in self.mlp_q_layers:
            # 使用 Xavier 初始化权重
            size = layer.weight.size()  # 获取层权重的尺寸
            fan_out = size[0]  # 输出节点数
            fan_in = size[1]  # 输入节点数
            std = np.sqrt(2.0 / (fan_in + fan_out))  # 计算标准差
            layer.weight.data.normal_(0.0, std)  # 用正态分布初始化权重
            # 正态分布初始化偏置
            layer.bias.data.normal_(0.0, 0.001)  # 偏置初始化为接近0的值

        for layer in self.mlp_p1_1_layers:
            # 使用 Xavier 初始化权重
            size = layer.weight.size()  # 获取层权重的尺寸
            fan_out = size[0]  # 输出节点数
            fan_in = size[1]  # 输入节点数
            std = np.sqrt(2.0 / (fan_in + fan_out))  # 计算标准差
            layer.weight.data.normal_(0.0, std)  # 用正态分布初始化权重
            # 正态分布初始化偏置
            layer.bias.data.normal_(0.0, 0.001)  # 偏置初始化为接近0的值

        for matrix in self.mlp_p1_2_layers:
            # 使用 Xavier 初始化权重
            size = matrix.data.size()  # 获取矩阵的尺寸
            fan_out = size[0]  # 输出节点数
            fan_in = size[1]  # 输入节点数
            std = np.sqrt(2.0 / (fan_in + fan_out))  # 计算标准差
            matrix.data.normal_(0.0, std)  # 用正态分布初始化权重

        for layer in self.mlp_p2_layers:
            # 使用 Xavier 初始化权重
            size = layer.weight.size()  # 获取层权重的尺寸
            fan_out = size[0]  # 输出节点数
            fan_in = size[1]  # 输入节点数
            std = np.sqrt(2.0 / (fan_in + fan_out))  # 计算标准差
            layer.weight.data.normal_(0.0, std)  # 用正态分布初始化权重
            # 正态分布初始化偏置
            layer.bias.data.normal_(0.0, 0.001)  # 偏置初始化为接近0的值

        for layer in self.mlp_p3_layers:
            # 使用 Xavier 初始化权重
            size = layer.weight.size()  # 获取层权重的尺寸
            fan_out = size[0]  # 输出节点数
            fan_in = size[1]  # 输入节点数
            std = np.sqrt(2.0 / (fan_in + fan_out))  # 计算标准差
            layer.weight.data.normal_(0.0, std)  # 用正态分布初始化权重
            # 正态分布初始化偏置
            layer.bias.data.normal_(0.0, 0.001)  # 偏置初始化为接近0的值

    def reg_loss(self):
        r"""计算模型参数的 L2 正则化损失。
        包括嵌入矩阵和模型的权重矩阵。
        返回：
            loss(torch.FloatTensor): L2 损失张量，形状为 [1,]
        """
        reg_loss = 0  # 初始化正则化损失
        for name, parm in self.mlp_q_layers.named_parameters():
            if name.endswith('weight'):  # 仅对权重进行正则化
                reg_loss = reg_loss + self.regs * (1 / 2) * parm.norm(2).pow(2)  # 计算 L2 范数的平方并累加
        for name, parm in self.mlp_p1_1_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1 / 2) * parm.norm(2).pow(2)
        for name, parm in self.mlp_p2_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1 / 2) * parm.norm(2).pow(2)
        for name, parm in self.mlp_p3_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1 / 2) * parm.norm(2).pow(2)
        return reg_loss  # 返回最终的正则化损失

def loss_function(recon_x, x, mu, logvar, anneal=1.0):
    # 计算重构误差（二元交叉熵损失）
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    # 计算 Kullback-Leibler 散度损失
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    # 返回总损失（重构误差 + KLD）
    return BCE + anneal * KLD