import logging
import os

import torch
from tqdm import tqdm

import utils
from evaluate import evaluate_epoch
from models.model import NERTokenClassification
from utils import Params, RunningAverage


def train_epoch(model: NERTokenClassification, dataloader, optimizer, params: Params):
    """
    一个epoch的训练
    :param model: 模型对象
    :param dataloader: 数据迭代器
    :param optimizer: 优化器
    :param params: 参数对象
    :return:
    """
    device = params.device
    # 设置模型为训练阶段
    model.train()
    # 遍历批次
    step = 1
    bar = tqdm(dataloader)
    loss_avg = RunningAverage()  # 记录平均损失
    for input_ids, input_mask, label_ids in bar:
        # to --> device
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        label_ids = label_ids.to(device)
        # 计算损失
        loss, _ = model(input_ids, input_mask, label_ids)
        # todo：多gpu运行的时候模型返回的是一个tensor[N]结构
        if params.n_gpu > 1:
            loss = loss.mean()

        # 更新间隔批次大于1时， 梯度累加相当于损失平均
        if params.gradient_accumulation_steps > 1:
            loss = loss / params.gradient_accumulation_steps
        # 反向传播-求解梯度
        loss.backward()
        # 仅且只当 1 % 1 == 0 时，一个批次内参数进行更新，梯度清零，参数更新的批次间隔大于1时，梯度不清零
        if step % params.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        # 日志信息描述
        loss = loss.item() * params.gradient_accumulation_steps  # 返回真实的当前批次的损失
        loss_avg.update(loss)   # 训练一个批次做一次平均损失
        bar.set_postfix(ordered_dict={
            'batch_loss': f'{loss:.3f}',  # 当前批次的额损失
            'loss': f'{loss_avg():.3f}'  # 一个epoch的动态平均损失
        })
        step += 1

def train_and_evaluate(model: NERTokenClassification, optimizer, train_loader, val_loader, params: Params):
    """训练所有批次"""
    # 参数持久化保存
    params.save_params()
    # 训练&验证
    best_val_f1 = 0.0
    stop_counter = 0
    # 断点恢复
    start_epoch = 1
    if os.path.exists(params.models_path) and len(os.listdir(params.models_path)) > 0:
        ckpt = torch.load(os.path.join(params.models_path, 'last.pth'), map_location='cpu')
        # 模型恢复
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1

    for epoch in range(start_epoch, params.epoch_num + 1):
        logging.info(f"Epoch {epoch}/{params.epoch_num}")
        # Train model
        train_epoch(model, train_loader, optimizer, params)
        # Evaluate模型效果 todo:未看！
        val_metrics = evaluate_epoch(model, val_loader, params, mark='Val', verbose=True)
        val_f1 = val_metrics['f1']  # 验证集的f1指标
        improve_val_f1 = val_f1 - best_val_f1  # 当前epoch训练后，验证集的f1指标提升值

        # 模型保存
        utils.save_checkpoint(
            state={
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            },
            is_best=improve_val_f1 > 0.0,
            checkpoint=params.models_path
        )
        # 提前停止训练的一个判断
        if improve_val_f1 > 0:
            logging.info(f"-- Find a better model, Val f1:{val_f1:.3f}")
            best_val_f1 = val_f1
            if improve_val_f1 <= params.stop_improve_val_f1_threshold:
                stop_counter += 1  # 当前epoch模型提升效果不明显，epoch停止计数器+1
            else:
                stop_counter = 0
        else:
            stop_counter += 1  # 当前epoch模型效果没有提升
        best_epoch = epoch - stop_counter  # 记录当前最好模型的训练批次
        logging.info(f'-- Best model at epoch:{best_epoch}, best val f1:{best_val_f1:.3f}')
        # 当效果不明显的epoch次数超过阈值
        # 且达到了模型要求的epoch训练次数
        if stop_counter > params.stop_epoch_num_threshold and epoch > params.min_epoch_num_threshold:
            logging.info(f"Early stop model training:{epoch}.")
            break
        if epoch == params.epoch_num:
            logging.info(f"Best model at the final epoch!")
            break

