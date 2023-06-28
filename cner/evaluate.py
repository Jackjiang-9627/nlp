import logging

import torch
import torch.nn as nn
from tqdm import tqdm

from metrics import f1_score, accuracy_score, classification_report
from models.model import NERTokenClassification
from utils import Params, RunningAverage


@torch.no_grad()
def evaluate_epoch(model: NERTokenClassification, data_loader, params: Params, mark='Val', verbose=True):
    device = params.device
    # 模型进入验证阶段
    model.eval()
    # id到tag的映射
    idx2tag = {idx: tag for tag, idx in params.tag2idx.items()}
    # 遍历数据，获取loss和预测标签值
    loss_avg = RunningAverage()
    pred_tags = []  # 预测标签 list[str]
    true_tags = []  # 真实标签 list[str]
    for input_ids, input_mask, label_ids in tqdm(data_loader):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        label_ids = label_ids.to(device)
        batch_size, max_len = label_ids.size()

        # loss & inference
        # return_output = True  返回预测结果
        loss, batch_output = model(input_ids, input_mask, labels=label_ids, return_output=True)
        # todo: device to gpu
        if params.n_gpu > 1:
            loss = loss.mean()  # 多gpu运行的时候模型返回的是一个tensor[N]结构
        loss_avg.update(loss.item())

        # 恢复真实标签的信息
        real_batch_label_ids = []
        for i in range(batch_size):
            real_len = input_mask[i].sum()
            # label_ids填充过，需要对其去除填充值
            real_batch_label_ids.append(label_ids[i][:real_len].to('cpu').numpy())
        # 将id转换为tag  --> List[int]
        pred_tags.extend(
            [idx2tag.get(idx) for indices in batch_output for idx in indices]
        )
        true_tags.extend(
            [idx2tag.get(idx) for indices in real_batch_label_ids for idx in indices]
        )
    assert len(pred_tags) == len(true_tags), 'len(pred_tags) is not equal to len(true_tags)!'

    # 开始计算评估指标
    metrics = {
        'loss': loss_avg(),  # 一个epoch内的动态损失
        'f1': f1_score(true_tags, pred_tags),  # todo：如何计算多分类的f1指标？
        'accuracy': accuracy_score(true_tags, pred_tags)
    }
    logging.info(f"-{mark} metrics: {'; '.join(map(lambda t: f'{t[0]}: {t[1]:.3f}', metrics.items()))}")
    if verbose:  # 显示详细信息
        report = classification_report(true_tags, pred_tags)
        logging.info(report)
    return metrics
