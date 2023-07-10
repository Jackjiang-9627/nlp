import logging
import logging.config
import os

import numpy as np
import torch
from torch import optim, nn
from tqdm import tqdm

from data_helper import load_train_eval_dataloader, fetch_tokens_from_text
from metrics import accuracy
from models.model import TextClassifyModel
from utils import create_dir

LOGGING_NAME = 'log'


def set_logging(name=LOGGING_NAME, verbose=True):
    # sets up logging for the given name
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            name: {
                "format": "%(message)s"}},
        "handlers": {
            name: {
                "class": "logging.StreamHandler",
                "formatter": name,
                "level": level, }},
        "loggers": {
            name: {
                "level": level,
                "handlers": [name],
                "propagate": False, }}})


set_logging(LOGGING_NAME)
LOGGER = logging.getLogger(LOGGING_NAME)

class Trainer(object):
    def __init__(self, token_vocab_file, label_vocab_file, output_dir):
        """文件路径相关参数，创建文件输出目录"""
        super(Trainer, self).__init__()
        # 加载数据
        self.token_vocab = torch.load(token_vocab_file, map_location='cpu')
        self.label_vocab = torch.load(label_vocab_file, map_location='cpu')
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 文件输出
        self.output_model_dir = os.path.join(output_dir, "models")
        self.output_summary_dir = os.path.join(output_dir, "summary")
        create_dir(self.output_model_dir)  # 模型保存路径
        create_dir(self.output_summary_dir)  # summary文件保存路径

    def train_epoch(self, epoch, batch_step, dataloader, model, loss_fn, opt, log_interval_batch, writer):
        # 进入训练阶段
        model.train()
        losses = []
        accuracies = []
        for _, _, batch_seq_idxes, _, batch_seq_length, _, batch_label_id in dataloader:
            batch_seq_idxes = batch_seq_idxes.to(self.device)
            batch_seq_length = batch_seq_length.to(self.device)
            batch_label_id = batch_label_id.to(self.device)

            # 前向过程
            score = model(batch_seq_idxes, batch_seq_length)
            loss = loss_fn(score, batch_label_id.long())
            acc = accuracy(score, batch_label_id)
            losses.append(loss.item())
            accuracies.append(acc.item())

            # 反向过程
            opt.zero_grad()
            loss.backward()
            opt.step()

            # 间隔某个batch，打印一次日志
            if batch_step % log_interval_batch == 0:
                writer.add_scalar('batch_train_loss', loss, global_step=batch_step)
                writer.add_scalar('batch_train_acc', acc, global_step=batch_step)
                LOGGER.info(f"Epoch:{epoch} Train Batch:{batch_step}  Loss:{loss.item():.3f} Accuracy:{acc.item():.3f}")
            batch_step += 1
        writer.add_scalars('epoch_train', {'loss': np.mean(losses), 'acc': np.mean(accuracies)}, global_step=epoch)
        return batch_step

    def eval_epoch(self, epoch, batch_step, dataloader, model, loss_fn, log_interval_batch, writer):
        # 进行评估推理阶段
        model.eval()
        losses = []
        accuracies = []
        for _, _, batch_seq_idxes, _, batch_seq_length, _, batch_label_id in dataloader:
            batch_seq_idxes = batch_seq_idxes.to(self.device)
            batch_seq_length = batch_seq_length.to(self.device)
            batch_label_id = batch_label_id.to(self.device)

            # 前向过程
            score = model(batch_seq_idxes, batch_seq_length)
            loss = loss_fn(score, batch_label_id.long())
            acc = accuracy(score, batch_label_id)

            # 打印日志
            if batch_step % log_interval_batch == 0:
                writer.add_scalar('batch_eval_loss', loss, global_step=batch_step)
                writer.add_scalar('batch_eval_acc', acc, global_step=batch_step)
                LOGGER.info(f"Epoch:{epoch} Eval Batch:{batch_step}  Loss:{loss.item():.3f} Accuracy:{acc.item():.3f}")
            batch_step += 1
        writer.add_scalars('epoch_eval', {'loss': np.mean(losses), 'acc': np.mean(accuracies)}, global_step=epoch)
        return batch_step

    def save_model(self, epoch, model, optimizer):
        obj = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(obj, os.path.join(self.output_model_dir, f"model_{epoch:06d}.pkl"))

    def training(self, cfg, data_files, n_epochs, batch_size, eval_ratio, save_interval_epoch, log_interval_batch,
                 ckpt_path=None, lr=0.01, weight_decay=0.0):
        """
        训练
        :param cfg: 模型配置dict字典对象
        :param data_files: 数据文件路径
        :param n_epochs: 训练的epoch的数量
        :param batch_size: 批次大小
        :param eval_ratio: 验证数据集的占比
        :param save_interval_epoch: 间隔多少个epoch进行一次模型保存
        :param log_interval_batch: 间隔多少个batch进行一次日志打印
        :param ckpt_path: 模型恢复的路径
        :param lr: 学习率
        :param weight_decay: 惩罚性系数
        :return:
        """
        # 1. 加载数据
        train_dataloader, eval_dataloader = load_train_eval_dataloader(
            data_files=data_files,
            token_vocab=self.token_vocab,
            label_vocab=self.label_vocab,
            batch_size=batch_size,
            eval_ratio=eval_ratio
        )

        # 2. 参数恢复  构建相关模型
        weights = None
        model = TextClassifyModel.build_model(cfg, weights)

        # opt = optim.SGD(params=model.parameters(), lr=0.1, weight_decay=0.01)
        start_epoch = 0
        if ckpt_path is None:
            ckpts = os.listdir(self.output_model_dir)
            if len(ckpts) > 0:
                # 默认排序是升序，取最后一个epoch的模型，也可以只保存最后一个，这样不用排序选择
                ckpts = sorted(ckpts, key=lambda name: int(name.split(".")[0].split("_")[1]))
                ckpt_path = os.path.join(self.output_model_dir, ckpts[-1])
        if ckpt_path is not None and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(ckpt['model'])
            # optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt['epoch'] + 1
            # n_epochs = n_epochs + start_epoch

        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        optimizer = optim.AdamW(params=model.parameters(), lr=lr, weight_decay=weight_decay)
        LOGGER.info(f"{model}")
        loss_fn = nn.CrossEntropyLoss()

        # 3. 训练相关代码
        from torch.utils.tensorboard import SummaryWriter

        with SummaryWriter(log_dir=self.output_summary_dir) as writer:
            # 输出执行图
            writer.add_graph(model, [torch.randint(0, 10, (8, 10)), torch.randint(1, 10, (8,))])
            train_batch_step, eval_batch_step = 0, 0
            for epoch in range(start_epoch, n_epochs):
                # 训练
                train_batch_step = self.train_epoch(
                    epoch, train_batch_step, train_dataloader, model, loss_fn, optimizer, log_interval_batch, writer
                )
                # 评估
                eval_batch_step = self.eval_epoch(epoch, eval_batch_step, eval_dataloader, model, loss_fn,
                                                  log_interval_batch, writer)
                # 模型保存
                if epoch % save_interval_epoch == 0:
                    self.save_model(epoch, model, optimizer)
                    # torch.save(model, "./datas/whole_model.pth")
            # 最终模型保存
            self.save_model(n_epochs, model, optimizer)
            LOGGER.info("训练完成!!")

class Evaluator(object):
    def __init__(self, cfg, token_vocab_file, label_vocab_file, output_dir, ckpt_path=None):
        super(Evaluator, self).__init__()

        # 加载数据
        self.token_vocab = torch.load(token_vocab_file, map_location='cpu')
        self.label_vocab = torch.load(label_vocab_file, map_location='cpu')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_model_dir = os.path.join(output_dir, "models")
        self.eval_output_dir = os.path.join(output_dir, "eval")
        create_dir(self.eval_output_dir)

        # 2. 参数恢复
        weights = None
        if ckpt_path is None:
            ckpts = os.listdir(self.output_model_dir)
            if len(ckpts) > 0:
                ckpts = sorted(ckpts, key=lambda name: int(name.split(".")[0].split("_")[1]))  # 默认排序是升序
                ckpt_path = os.path.join(self.output_model_dir, ckpts[-1])
        if ckpt_path is not None and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            weights = ckpt['model']
        if weights is None:
            raise ValueError(f"无法进行模型恢复:{self.output_model_dir} -- {ckpt_path}")

        # 恢复模型
        cfg['token_emb_layer']['args'].insert(0, len(self.token_vocab))
        cfg['classify_decision_layer']['args'].insert(0, len(self.label_vocab))
        self.model = TextClassifyModel.build_model(cfg, weights, strict=True).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()

    @torch.no_grad()
    def eval(self, data_files, batch_size):
        # 1. 加载数据
        _, eval_dataloader = load_train_eval_dataloader(
            data_files=data_files,
            token_vocab=self.token_vocab,
            label_vocab=self.label_vocab,
            batch_size=batch_size,
            eval_ratio=1.0
        )

        # 2. 遍历数据进行评估
        targets = []
        scores = []
        self.model.eval()  # 评估推理阶段
        with open(os.path.join(self.eval_output_dir, "eval.txt"), "w", encoding='utf-8') as writer:
            for batch_text, batch_tokens, _, batch_seq_idxes, batch_seq_length, batch_label, batch_label_id in tqdm(
                    eval_dataloader):
                batch_seq_idxes = batch_seq_idxes.to(self.device)
                batch_seq_length = batch_seq_length.to(self.device)
                batch_label_id = batch_label_id.to(self.device)

                score = self.model(batch_seq_idxes, batch_seq_length)
                scores.append(score)
                targets.append(batch_label_id)
                predict_label_id = torch.argmax(score, dim=1)  # [N,num_classes] -> [N]

                bs = len(batch_text)
                predict_label_id = predict_label_id.to('cpu').detach().numpy()
                predict_label = self.label_vocab.lookup_tokens(list(predict_label_id))
                batch_seq_idxes = batch_seq_idxes.to('cpu').detach().numpy()
                for i in range(bs):
                    msg = f"{batch_text[i]} | {batch_tokens[i]} | {list(batch_seq_idxes[i])} | {batch_label[i]} | {predict_label[i]}"
                    writer.writelines(f"{msg}\n")

        score = torch.concat(scores, dim=0)
        target = torch.concat(targets, dim=0)
        acc = accuracy(score, target)
        loss = self.loss_fn(score, target.long())
        LOGGER.info(f"Accuracy:{acc.item():.4f} Loss:{loss.item():.4f}")


class Predictor(object):
    """要求：恢复模型、针对给定的文本返回对应的预测类别以及概率值、支持topk返回"""
    def __init__(self, ckpt_path, token_vocab_file, label_vocab_file, num_token='<NUM>', pun_token='<PUN>'):
        # 1. 模型恢复
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.model = ckpt['model']
        self.model.eval()  # 进行推理阶段
        self.model.to(self.device)

        # 2. 映射字典恢复
        self.token_vocab = torch.load(token_vocab_file, map_location='cpu')
        self.label_vocab = torch.load(label_vocab_file, map_location='cpu')
        self.num_token = num_token
        self.pun_token = pun_token

    @torch.no_grad()
    def predict(self, text: str, k: int = 1, probability_threshold: float = 0.1, is_debug: bool = False):
        """
        针对给定文本返回对应的预测结果
        :param text: 文本字符串,eg:"今天给我一部张艺谋拍的纪录片看一看"
        :param k: 是否进行top-k预测，返回K个概率最高的预测结果
            k==1: [('FilmTele-Play', 0.72)]
            k==2: [('FilmTele-Play', 0.72), ['Video-Play', 0.21]]
        :param probability_threshold: 概率阈值，仅返回预测概率大于等于该值的类别
        :param is_debug: 是不是debug模式
        :return: 数组，数组内按照预测概率降序排列的预测结果 (预测类别，预测概率)
        """
        # 文本转token
        tokens = fetch_tokens_from_text(text, self.num_token, self.pun_token)
        # token转id list序列（包括T个token id）
        token_idxes = self.token_vocab(tokens)
        if is_debug:
            print(text)
            print(tokens)
            print(token_idxes)
            print(self.token_vocab.lookup_tokens(token_idxes))
        # 调用模型
        seq_idxes = torch.tensor([token_idxes], dtype=torch.int32)  # [1, T]
        seq_lengths = torch.tensor([len(token_idxes)], dtype=torch.int32)  # [1]
        output_scores = self.model(seq_idxes, seq_lengths)  # [1, num_classes]
        output_probability = torch.softmax(output_scores, dim=1)  # [1, num_classes]
        topk_probability, topk_label_idxes = torch.topk(output_probability[0], k, dim=0, largest=True, sorted=True)
        # tensor结果转换numpy
        topk_probability = topk_probability.to('cpu').detach().numpy()
        topk_label_idxes = topk_label_idxes.to('cpu').detach().numpy()
        topk_labels = self.label_vocab.lookup_tokens(list(topk_label_idxes))
        # 结果拼接返回
        result = []
        for prob, label in zip(topk_probability, topk_labels):
            if prob < probability_threshold:
                continue
            result.append({'probability': float(f"{prob:.4f}"), 'class_name': label})
        return result
