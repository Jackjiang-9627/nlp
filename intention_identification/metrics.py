import torch


def accuracy(score, target):
    """
    准确率的计算
    :param score: 置信度，也就是模型前向输出的结果，[N,num_classes]
    :param target: 实际标签下标, [N]
    :return:
    """
    # 选置信度大的id作为预测标签
    predict_label = torch.argmax(score, dim=1).to(target.dtype)
    corr = (predict_label == target).to(torch.float)
    acc = torch.mean(corr)
    return acc
