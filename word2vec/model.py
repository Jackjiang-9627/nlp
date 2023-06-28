"""定义模型相关代码"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def random_negative_labels(positive_labels, num_negative_label, all_labels, negative_over_positive, all_label_weights):
    """
    随机产生负样本的标签
    :param all_label_weights: 所有标签的权重信息
    :param negative_over_positive: 产生负样本标签的时候，产生的数量是否加上正样本的标签数量
    :param all_labels: 所有单词的标签id列表
    :param positive_labels: 正样本的标签id
    :param num_negative_label: 负样本标签数量
    :return: 最终产生的负样本标签数量在num_negative_label附近即可，返回一个list[int/long]数组即可
    """
    positive_labels = np.asarray(positive_labels).reshape(-1)
    random_labels = np.random.choice(
        all_labels,
        size=num_negative_label + (len(positive_labels) if negative_over_positive else 0),
        p=all_label_weights
    )
    negative_labels = []
    for label in random_labels:
        if label in positive_labels:
            continue
        negative_labels.append(label)
    if len(negative_labels) == 0:
        # 如果随机采样全部为正样本的标签，那么这里直接等比例抽取样本（不考虑任何的样本重复问题）
        negative_labels = list(np.random.choice(all_labels, num_negative_label))
    return negative_labels


class Word2Vec(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, k=5, all_label_weights=None):
        """
        初始化方法
        :param num_embeddings: 词汇表大小
        :param embedding_dim: 词向量维度大小
        :param k: 负采样的样本大小
        :param all_label_weights: 所有标签的权重信息
        """
        super(Word2Vec, self).__init__()
        self.vocab_size = num_embeddings
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim)))

        # embedding层和fc层的参数矩阵来自同一个table
        self.emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        self.emb_layer.weight = self.weight
        # 参数初始化
        nn.init.normal_(self.weight)
        self.k = k
        self.loss_fn = nn.CrossEntropyLoss()
        self.all_labels = np.arange(self.vocab_size)
        # 正常情况下，这个所有标签/单词对应的权重就是单词的词频，一般情况下外部统计
        self.all_label_weights = all_label_weights
        self.negative_over_positive = False  # 产生负样本标签的时候，产生的数量是否加上正样本的标签数量


class CBOWModule(Word2Vec):
    def forward_with_negative_sampling(self, x: torch.tensor, y: torch.tensor):
        """
        :param x: [N, M] N 表示批次大小，M表示CBOW结构的窗口大小
        :param y: [N] N 表示批次大小，内部存储的是每个样本对应的中心词的id
        :return:
        """
        # 对参数weight进行l2-norm转换
        norm_weight = F.normalize(self.weight, p=2, dim=1)

        # embedding layer
        # x = self.emb_layer(x)  # [N, M] --> [N, M, E], 也就是每个token转换为E维向量
        x = norm_weight[x]  # 两者都可以

        # 合并M个token的向量
        x = torch.sum(x, dim=1)  # [N, M, E] --> [N, E]

        # 负采样
        # 将标签转换为正样本标签
        positive_labels = y.detach().numpy()
        # 随机选择k个值（不等比例产生数据）
        negative_labels = random_negative_labels(
            positive_labels, self.k, self.all_labels,
            self.negative_over_positive, self.all_label_weights
        )
        # 从参数weight中获取正样本和负样本对应的参数值
        positive_weights = norm_weight[positive_labels]  # [N, E]  -->每个正样本对应的权重
        negative_weights = norm_weight[negative_labels]  # [?k, E] --> ?K：负样本数量，每个负样本对应的权重
        # 计算正样本的置信度
        positive_scores = torch.sum(x * positive_weights, dim=1, keepdim=True)  # [N, E] * [N, E] --> [N, E] -->[N, 1]
        negative_scores = torch.matmul(x, negative_weights.T)  # [N, E] @ [E, ?k] --> [N, ?K]
        # 合并置信度, 第一列就是正样本的置信度
        scores = torch.concat([positive_scores, negative_scores], dim=1)  # [N, 1] [N, ?k] --> [N, 1+?k]
        # 构建“实际标签”
        target = torch.zeros(scores.shape[0], dtype=torch.long)
        # 计算损失
        loss = self.loss_fn(scores, target)
        return loss

    def forward_without_negative_sampling(self, x, y):
        """
        :param x: [N, M] N 表示批次大小，M表示CBOW结构的窗口大小
        :param y: [N] N 表示批次大小，内部存储的是每个样本对应的中心词的id
        :return:
        """
        # 对参数weight进行l2-norm转换
        norm_weight = F.normalize(self.weight, p=2, dim=1)

        # embedding layer提取每个单词对应的特征向量
        x = norm_weight[x]  # [N,M] --> [N,M,E] 也就是每个token转换成E维的向量
        # 合并M个token的向量
        x = torch.sum(x, dim=1)  # [N,M,E] --> [N,E]

        # fc全连接
        scores = F.linear(x, norm_weight)  # [N,E],[Vocab_size,E] --> [N,Vocab_size]

        # 损失计算
        loss = self.loss_fn(scores, y)
        return loss

    def forward(self, context_words, central_words, negative_sampling=True):
        if negative_sampling:
            return self.forward_with_negative_sampling(x=context_words, y=central_words)
        else:
            return self.forward_without_negative_sampling(x=context_words, y=central_words)


class SkipGramModule(CBOWModule):
    def forward_with_negative_sampling(self, x: torch.tensor, y: torch.tensor):
        """
        :param x: [N] N表示批次大小，内部存储的就是N个样本每个样本对应的中心词id
        :param y: [N,M] N表示批次大小(也就是一个批次中的样本数目), M表示结构中的窗口大小(也就是上下文的单词/token数目)
        :return:
        """
        # 对参数weight进行l2-norm转换
        norm_weight = F.normalize(self.weight, p=2, dim=1)

        # embedding layer提取每个单词对应的特征向量
        x = norm_weight[x]  # [N] --> [N,E] 也就是每个token转换成E维的向量

        ## 负采样
        # 将标签转换为正样本标签
        positive_labels = y.detach().numpy()  # [N,M] numpy类型
        # 随机选择k个值(不等比率的产生数据)
        negative_labels = random_negative_labels(
            positive_labels, self.k, self.all_labels, self.all_label_weights, self.negative_over_positive
        )
        # 从参数weight中获取正样本和负样本对应的参数值
        positive_weights = norm_weight[y]  # [N,M,E] --> 每个正样本对应的权重，N表示N个样本，M表示每个样本有M个预测词，E表示每个预测词对应的E维的参数向量/权重
        negative_weights = norm_weight[negative_labels]  # [?k,E] --> ?k就是负样本的数量，也就是每个负样本对应的参数向量/权重
        # 计算正样本置信度
        # [N, E] --> [N, 1, E] * [N, M, E] --> [N, M, E] --> [N, M, 1]
        positive_scores = torch.sum(x[:, None, :] * positive_weights, dim=-1, keepdim=True)
        N, M, _ = positive_scores.shape
        negative_scores = x @ negative_weights.T  # [N, E]@[E, ?k] --> [N, ?k]
        negative_scores = torch.tile(negative_scores[:, None, :], (1, M, 1))  # [N, ?k] --> [N, M, ?k]
        # 正负样本置信度合并
        scores = torch.concat((positive_scores, negative_scores), dim=-1)  # --> [N, M, 1+?k]
        scores = torch.transpose(scores, 1, 2)  # --> [N, 1+?k, M]
        # 构建“实际标签”
        target = torch.zeros((N, M), dtype=torch.long)
        # 计算损失
        loss = self.loss_fn(scores, target)
        return loss

    def forward_without_negative_sampling(self, x, y):
        """
        :param x: [N] N表示批次大小，内部存储的就是N个样本每个样本对应的中心词id
        :param y: [N,M] N表示批次大小(也就是一个批次中的样本数目), M表示结构中的窗口大小(也就是上下文的单词/token数目)
        :return:
        """
        # 对参数weight进行l2-norm转换
        norm_weight = F.normalize(self.weight, p=2, dim=1)

        # embedding layer提取每个单词对应的特征向量
        x = norm_weight[x]  # [N] --> [N,E] 也就是每个token转换成E维的向量

        # fc全连接
        scores = F.linear(x, norm_weight)  # [N,E],[Vocab_size,E] --> [N,Vocab_size]
        _, M = y.shape
        scores = torch.tile(scores[..., None], (1, 1, M))  # [N,Vocab_size] --> [N,Vocab_size,1] --> [N,Vocab_size,M]

        # 损失构建
        loss = self.loss_fn(scores, y)
        return loss

    def forward(self, context_words, central_words, negative_sampling=True):
        if negative_sampling:
            return self.forward_with_negative_sampling(x=central_words, y=context_words)
        else:
            return self.forward_without_negative_sampling(x=central_words, y=context_words)


if __name__ == '__main__':
    context_word = torch.tensor([
        [2, 3, 4, 5, 6],
        [1, 6, 4, 2, 1],
        [4, 2, 4, 2, 1]
    ])
    center_word = torch.tensor([9, 8, 0])
    model = CBOWModule(100, 4)
    loss = model(context_word, center_word, negative_sampling=True)
    print(loss)
    # model = SkipGramModule(100, 4)
    # loss = model(center_word, context_word, negative_sampling=True)
    # print(loss)
