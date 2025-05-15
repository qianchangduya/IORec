import numpy as np
import torch  # 导入PyTorch库
import math  # 导入数学库
import time  # 导入时间库


def pre_ranking(user_feature, item_feature, train_dict, valid_dict, test_dict):
    '''准备排名：构造输入数据'''

    user_rank_feature = {}  # 初始化用户排名特征字典
    for userID in test_dict:  # 遍历测试集中的每个用户
        his_items = train_dict[userID]  # 获取该用户的历史物品
        features = []  # 特征列表
        feature_values = []  # 特征值列表
        mask = []  # 掩码列表
        item_idx = list(item_feature.keys())  # 获取所有物品的索引
        for idx in range(len(item_idx)):  # 遍历所有物品
            itemID = item_idx[idx]
            if itemID in his_items:  # 如果物品在训练集中，则掩码设置为-999
                mask.append(-999.0)
            else:  # 否则掩码设置为0
                mask.append(0.0)
            # 将用户特征和物品特征合并
            features.append(np.array(user_feature[userID][0] + item_feature[itemID][0]))
            feature_values.append(np.array(user_feature[userID][1] + item_feature[itemID][1], dtype=np.float32))

        # 将特征和特征值转换为PyTorch张量并移动到GPU
        features = torch.tensor(features).cuda()
        feature_values = torch.tensor(feature_values).cuda()
        mask = torch.tensor(mask).cuda()
        user_rank_feature[userID] = [features, feature_values, mask]  # 存储用户的特征、特征值和掩码

    return user_rank_feature  # 返回用户排名特征


def Ranking(model, valid_dict, test_dict, train_dict, item_feature, user_rank_feature, \
            batch_size, topN, return_pred=False):
    """通过召回率、精确度和NDCG评估Top-N排名的性能"""
    user_gt_test = []  # 用户测试集的真实标签
    user_gt_valid = []  # 用户验证集的真实标签
    user_pred = []  # 用户预测的物品
    user_pred_dict = {}  # 用户预测字典
    user_item_top1k = {}  # 用户推荐的Top-1K物品

    for userID in test_dict:  # 遍历测试集中的每个用户
        features, feature_values, mask = user_rank_feature[userID]  # 获取用户的特征、特征值和掩码

        batch_num = len(item_feature) // batch_size  # 计算批次数
        item_idx = list(item_feature.keys())  # 获取物品索引
        st, ed = 0, batch_size  # 初始化起始和结束索引

        for i in range(batch_num):  # 遍历每个批次
            batch_feature = features[st: ed]  # 获取当前批次的特征
            batch_feature_values = feature_values[st: ed]  # 获取当前批次的特征值
            batch_mask = mask[st: ed]  # 获取当前批次的掩码

            prediction = model(batch_feature, batch_feature_values)  # 使用模型进行预测
            prediction = prediction + batch_mask  # 将掩码添加到预测中
            if i == 0:
                all_predictions = prediction  # 保存第一次预测
            else:
                all_predictions = torch.cat([all_predictions, prediction], 0)  # 将后续批次的预测结果连接起来

            st, ed = st + batch_size, ed + batch_size  # 更新起始和结束索引

        # 处理最后一个批次的预测
        batch_feature = features[st:]  # 获取最后一个批次的特征
        batch_feature_values = feature_values[st:]  # 获取最后一个批次的特征值
        batch_mask = mask[st:]  # 获取最后一个批次的掩码

        prediction = model(batch_feature, batch_feature_values)  # 进行预测
        prediction = prediction + batch_mask  # 添加掩码
        if batch_num == 0:
            all_predictions = prediction  # 如果没有批次，直接保存预测
        else:
            all_predictions = torch.cat([all_predictions, prediction], 0)  # 否则连接预测结果

        user_gt_valid.append(valid_dict[userID])  # 保存验证集真实标签
        user_gt_test.append(test_dict[userID])  # 保存测试集真实标签
        _, indices = torch.topk(all_predictions, topN[-1])  # 获取Top-N预测物品的索引
        pred_items = torch.tensor(item_idx)[indices].cpu().numpy().tolist()  # 转换为物品ID
        user_item_top1k[userID] = pred_items  # 存储用户的Top-1K推荐物品
        user_pred_dict[userID] = all_predictions.detach().cpu().numpy()  # 存储用户的预测结果
        user_pred.append(pred_items)  # 将预测物品添加到列表中

    # 计算验证集和测试集的Top-N准确率
    valid_results = computeTopNAccuracy(user_gt_valid, user_pred, topN)
    test_results = computeTopNAccuracy(user_gt_test, user_pred, topN)

    if return_pred:  # 如果需要返回预测结果
        return valid_results, test_results, user_pred_dict, user_item_top1k
    return valid_results, test_results  # 返回验证和测试结果


def sigmoid(x):
    """计算sigmoid函数"""
    s = 1 / (1 + np.exp(-x))  # sigmoid公式
    return s


def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
    """计算Top-N的准确率、召回率、NDCG和MRR"""
    precision = []  # 精确度列表
    recall = []  # 召回率列表
    NDCG = []  # NDCG列表
    MRR = []  # MRR列表

    for index in range(len(topN)):  # 遍历每个topN
        sumForPrecision = 0  # 精确度累加器
        sumForRecall = 0  # 召回率累加器
        sumForNdcg = 0  # NDCG累加器
        sumForMRR = 0  # MRR累加器
        for i in range(len(predictedIndices)):  # 对于每个用户
            if len(GroundTruth[i]) != 0:  # 如果真实标签不为空
                mrrFlag = True  # 标记MRR
                userHit = 0  # 用户
                userMRR = 0  # 用户的平均倒排精度（Mean Reciprocal Rank）初始化为0
                dcg = 0  # 累计折现增益（Discounted Cumulative Gain）初始化为0
                idcg = 0  # 理想折现增益（Ideal Discounted Cumulative Gain）初始化为0
                idcgCount = len(GroundTruth[i])  # 理想相关物品的数量
                ndcg = 0  # NDCG初始化为0
                hit = []  # 记录命中的物品ID
                for j in range(topN[index]):  # 遍历每个Top-N
                    if predictedIndices[i][j] in GroundTruth[i]:  # 如果预测的物品在真实标签中命中
                        # 如果命中，计算DCG
                        dcg += 1.0 / math.log2(j + 2)  # 更新DCG值
                        if mrrFlag:  # 如果是第一次命中
                            userMRR = (1.0 / (j + 1.0))  # 更新用户的MRR
                            mrrFlag = False  # 设置标志为False，后续命中不再更新MRR
                        userHit += 1  # 用户命中的物品计数增加

                    if idcgCount > 0:  # 如果还有理想相关物品
                        idcg += 1.0 / math.log2(j + 2)  # 更新IDCG值
                        idcgCount -= 1  # 理想相关物品计数减少

                # 如果IDCG不等于0，计算NDCG
                if idcg != 0:
                    ndcg += (dcg / idcg)  # 计算NDCG

                # 更新精确度、召回率、NDCG和MRR的累计值
                sumForPrecision += userHit / topN[index]  # 更新精确度的总和
                sumForRecall += userHit / len(GroundTruth[i])  # 更新召回率的总和
                sumForNdcg += ndcg  # 更新NDCG的总和
                sumForMRR += userMRR  # 更新MRR的总和

        # 计算平均精确度、召回率、NDCG和MRR并保留四位小数
        precision.append(round(sumForPrecision / len(predictedIndices), 4))  # 计算并添加精确度
        recall.append(round(sumForRecall / len(predictedIndices), 4))  # 计算并添加召回率
        NDCG.append(round(sumForNdcg / len(predictedIndices), 4))  # 计算并添加NDCG
        MRR.append(round(sumForMRR / len(predictedIndices), 4))  # 计算并添加MRR

    return precision, recall, NDCG, MRR  # 返回精确度、召回率、NDCG和MRR



def print_results(loss, valid_result, test_result):
    """输出评价结果"""
    if loss is not None:  # 如果损失不为空
        print("[Train]: loss: {:.4f}".format(loss))  # 打印训练损失
    if valid_result is not None:  # 如果验证结果不为空
        print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in valid_result[0]]),  # 打印精确度
                            '-'.join([str(x) for x in valid_result[1]]),  # 打印召回率
                            '-'.join([str(x) for x in valid_result[2]]),  # 打印NDCG
                            '-'.join([str(x) for x in valid_result[3]])))  # 打印MRR
    if test_result is not None:  # 如果测试结果不为空
        print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in test_result[0]]),  # 打印测试集的精确度
                            '-'.join([str(x) for x in test_result[1]]),  # 打印测试集的召回率
                            '-'.join([str(x) for x in test_result[2]]),  # 打印测试集的NDCG
                            '-'.join([str(x) for x in test_result[3]])))  # 打印测试集的MRR

