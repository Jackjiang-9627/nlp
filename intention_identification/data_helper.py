"""数据处理相关代码"""
import re
import warnings
from typing import Iterator, List
import numpy as np
import torch
import jieba
import torchtext
from collections import OrderedDict

from torch.utils.data import Dataset, Sampler
from torch.utils.data.dataset import T_co

from intention_identification import get_project_absolute_path
# from . import get_project_absolute_path  # __main__中不能使用相对导入
from utils import create_file_dir, read_file

jieba.load_userdict(get_project_absolute_path('datas/jieba_words.txt'))  # 注意：此处应该用绝对路径！
# jieba.suggest_freq(['xx', 'xx'], True)
re_number = re.compile("([0-9\.]+)", re.U)
re_punctuation = re.compile("([,.，。？！：；、“\"]+)", re.U)


def split_text_token(text):
    return jieba.lcut(text)


def fetch_tokens_from_text(text, num_token, pun_token):
    tokens = []
    for token in split_text_token(text):
        # 判断是不是数字以及标点符号
        if is_number(token):
            token = num_token
        elif is_punctuation(token):
            token = pun_token
        tokens.append(token)
    return tokens


def split_data(in_file, out_file, num_token, pun_token, train_data=True):
    create_file_dir(out_file)
    with open(out_file, 'w', encoding='utf-8') as writer:
        for line in read_file(in_file):
            line = line.strip()
            if train_data:  # 去除训练集标签
                line = line.split('\t', maxsplit=2)[0]
            tokens = fetch_tokens_from_text(line, num_token, pun_token)
            writer.writelines(f"{' '.join(tokens)}\n")


def is_number(token):
    try:
        if re_number.match(token):
            return True
        return False
    except Exception as e:
        return False


def is_punctuation(token):
    try:
        if re_punctuation.match(token):
            return True
        return False
    except Exception as e:
        return False


def load_stop_tokens(in_file):
    """加载停止词"""
    tokens = []
    for line in read_file(in_file):
        tokens.append(line)
    # with open(in_file, 'r', encoding='utf-8') as reader:
    #     for line in reader:
    #         line = line.strip()
    #         tokens.append(line)
    return tokens


def build_vocab(in_files, out_file, stop_tokens, min_freq, default_tokens, unk_token):
    create_file_dir(out_file)
    # 1. 第一步构建dict字典
    token_dict = {}  # token以及对应token的词频
    for line in read_file(in_files):
        line.strip() 
        tokens = line.split(' ')
        for token in tokens:
            if token in stop_tokens:  # 过滤停止词
                continue
            token_dict[token] = token_dict.get(token, 0) + 1
    # 2. 去除词频小于参数的单词
    token_dict = {token: token_freq for token, token_freq in token_dict.items() if token_freq >= min_freq}
    token_dict = OrderedDict(token_dict)
    # 3. 进行torch词典对象构建
    vocab = torchtext.vocab.vocab(token_dict)
    for token, token_idx in default_tokens.items():
        if token in vocab:
            continue
        vocab.insert_token(token, token_idx)
    vocab.set_default_index(vocab[unk_token])
    # 4. 保存  转换静态script对象后，部分方法无法访问  在训练前，必须保存普通类型的对象
    # vocab = torch.jit.script(vocab)  # 转换为torch的script结构（静态化转换）
    # vocab.save(out_file)
    torch.save(vocab, out_file)
    print(len(token_dict))
    print(token_dict)


def extract_classify_labels(in_file, out_file):
    """
    提取具体的意图类别数据
    """
    create_file_dir(out_file)

    # 读取标签类型
    label_2_cnt = {}
    for line in read_file(in_file):
        line = line.strip()  # 去除首尾空格
        record = line.split('\t')
        if len(record) >= 2:
            label = record[-1]
            label_2_cnt[label] = label_2_cnt.get(label, 0) + 1
    label_2_cnt = sorted(label_2_cnt.items(), key=lambda t: -t[1])  # 按照标签数量，降序排列
    label_2_cnt = OrderedDict(label_2_cnt)
    # 构建映射词典并保存
    vocab = torchtext.vocab.vocab(label_2_cnt, min_freq=0)
    vocab.set_default_index(vocab['Other'])  # 映射关系中，如果不存在对应映射的时候，直接返回Other对应的标签
    # vocab = torch.jit.script(vocab)  # 因为转换静态script对象后，部分方法无法访问， 在训练前，必须保存普通类型的对象
    # vocab.save(out_file)
    torch.save(vocab, out_file)
    return list(label_2_cnt.keys())


class MySampler(Sampler[int]):
    def __init__(self, sample_indexes: List[int], shuffle=True):
        super(MySampler, self).__init__(None)
        self.sample_indexes = sample_indexes
        self.shuffle = shuffle

    def __len__(self):
        return len(self.sample_indexes)

    def __iter__(self) -> Iterator[int]:
        sample_indexes = self.sample_indexes
        if self.shuffle:
            np.random.shuffle(sample_indexes)
        yield from sample_indexes


class MyDataset(Dataset):
    # 数据从original加载
    def __init__(self, data_files, token_vocab, label_vocab, pad_token='<PAD>',
                 num_token='<NUM>', pun_token='<PUN>', has_labels=True):
        super(MyDataset, self).__init__()
        if isinstance(token_vocab, str):
            self.token_vocab = torch.load(token_vocab, map_location='cpu')
        else:
            self.token_vocab = token_vocab
        if isinstance(label_vocab, str):
            self.label_vocab = torch.load(label_vocab, map_location='cpu')
        else:
            self.label_vocab = label_vocab
        self.pad_token_idx = self.token_vocab[pad_token]
        self.has_labels = has_labels
        # 加载数据到内存中
        self.records = []
        for line in read_file(data_files):
            if has_labels:
                # 按照有标签的逻辑处理
                text, label = line.split('\t', maxsplit=2)
            else:
                # 按照没有标签的逻辑处理
                text, label = line, None
            # 文本处理
            tokens = fetch_tokens_from_text(text, num_token, pun_token)
            token_ides = self.token_vocab(tokens)
            new_tokens = self.token_vocab.lookup_tokens(token_ides)
            # 标签处理
            label_id = None if label is None else self.label_vocab[label]
            # 结果保存到临时内存中
            self.records.append((text, new_tokens, token_ides, len(token_ides), label, label_id))

    def __getitem__(self, index) -> T_co:
        return self.records[index]

    def __len__(self):
        return len(self.records)

    def clip_append_pad_idx(self, seq_idxes, seq_expect_length):
        seq_idxes = list(seq_idxes)  # 如果不复制将会改变seq_idxes的值（也就是collect_fn函数里的item[2]将会被填充）
        seq_actual_length = len(seq_idxes)
        if seq_actual_length > seq_expect_length:
            warnings.warn(f'不建议进行序列阶段操作：{seq_actual_length} -- {seq_expect_length}')
            return seq_idxes[:seq_expect_length]
        else:
            for i in range(seq_expect_length - seq_actual_length):
                seq_idxes.append(self.pad_token_idx)
            return seq_idxes


    def collect_fn(self, batch):
        # text: List(str), new_tokens:list(list(str)), token_ides:Tensor [N,M], reverse_token_ides:Tensor [N,M],
        # seq_lengths:Tensor [N],  label: List(str), label_id: Tensor[N],
        # N 批次大小
        # T：表示序列/文本特征长度，表示每个样本文本有T个特征/token信息；处理过程中需要进行填充，也就可能更存在实际序列长度是5，填充后长度为13的情况
        # seq_lengths:额外增加的实际长度特征，每个样本的实际序列长度
        # reverse_token_ides: 反向序列
        batch_text, batch_tokens, batch_seq_idxes, batch_reverse_seq_idxes, batch_seq_length = [], [], [], [], []
        batch_label, batch_label_id = [], []
        max_length = int(max([item[3] for item in batch]))
        for item in batch:
            batch_text.append(item[0])
            batch_tokens.append(item[1])
            batch_seq_idxes.append(self.clip_append_pad_idx(item[2], max_length))  # 进行填充后的新的id序列
            batch_reverse_seq_idxes.append(self.clip_append_pad_idx(item[2][::-1], max_length))  # 进行填充后的新的id序列
            batch_seq_length.append(item[3])
            batch_label.append(item[4])
            batch_label_id.append(item[5])
        # 转换成tensor对象
        batch_seq_length = torch.tensor(batch_seq_length, dtype=torch.int32)
        batch_reverse_seq_idxes = torch.tensor(batch_reverse_seq_idxes, dtype=torch.int32)
        batch_seq_idxes = torch.tensor(batch_seq_idxes, dtype=torch.int32)
        if self.has_labels:
            batch_label_id = torch.tensor(batch_label_id, dtype=torch.int32)

        return batch_text, batch_tokens, batch_seq_idxes, batch_reverse_seq_idxes, batch_seq_length, batch_label, batch_label_id

def load_train_eval_dataloader(data_files, token_vocab, label_vocab, batch_size, eval_batch_size=None, eval_ratio=0.1):
    dataset = MyDataset(
        data_files=data_files,
        token_vocab=token_vocab,
        label_vocab=label_vocab,
        has_labels=True
    )
    # 拆分训练集和验证集下标数据
    n = len(dataset)
    random_idxes = np.random.permutation(n)  # [0, n-1]的随机排列值
    train_idxes = list(random_idxes[int(n * eval_ratio):])
    eval_idxes = list(random_idxes[:int(n * eval_ratio)])

    # 分别构建dataloader
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collect_fn,
        sampler=MySampler(train_idxes)  # 给定数据下标生成器,  shuffle必须给定False！
    )
    eval_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size * 2 if eval_batch_size is None else eval_batch_size,
        shuffle=False,
        collate_fn=dataset.collect_fn,
        sampler=MySampler(eval_idxes)  # 给定数据下标生成器
    )
    return train_dataloader, eval_dataloader

def load_test_dataloader(data_files, token_vocab, label_vocab, batch_size):
    dataset = MyDataset(
        data_files=data_files,
        token_vocab=token_vocab,
        label_vocab=label_vocab,
        has_labels=False
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collect_fn
    )
    return test_dataloader


if __name__ == '__main__':
    # 打印类别数据
    # r1 = extract_classify_labels(
    #     in_file='datas/original/train.csv',
    #     out_file='datas/output/label_vocab.pkl'
    # )
    # print(len(r1))
    # print(r1)

    # import sys
    # print(sys.path)  # 导包时查看系统环境

    # 分词
    # split_data(in_file='datas/original/train.csv', out_file='datas/output/train_tokens.csv', train_data=True,
    #            num_token='<NUM>', pun_token='<PUN>')
    # split_data(in_file='datas/original/test.csv', out_file='datas/output/test_tokens.csv', train_data=False,
    #            num_token='<NUM>', pun_token='<PUN>')

    # 构建词典一般是所有数据一起构建（训练集和验证/测试集的token高度重合）
    # build_vocab(in_files=['datas/output/train_tokens.csv', 'datas/output/test_tokens.csv'],
    #             out_file='datas/output/vocab.pkl',
    #             stop_tokens=load_stop_tokens('datas/stop_words.txt'), min_freq=2,
    #             default_tokens={
    #                 '<PAD>': 0,  # 填充字符串
    #                 '<UNK>': 1  # 未知字符串
    #             },
    #             unk_token='<UNK>')

    token_vocab = torch.load('datas/output/vocab.pkl', map_location='cpu')
    label_vocab = torch.load('datas/output/label_vocab.pkl', map_location='cpu')

    train_dataset = MyDataset(
        data_files=['datas/original/train.csv'],
        token_vocab=token_vocab,
        label_vocab=label_vocab,
        has_labels=True
    )
    print(len(train_dataset))
    # test_dataset = MyDataset(
    #     data_files=['datas/original/test.csv'],
    #     token_vocab=token_vocab,
    #     label_vocab=label_vocab,
    #     has_labels=False
    # )
    # print(train_dataset[23])
    # print(train_dataset[56])
    #
    # test_dataloader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=4,
    #     shuffle=False,
    #     collate_fn=test_dataset.collect_fn
    # )

    traindataloader, evaldataloader = load_train_eval_dataloader(
        data_files=['datas/original/train.csv'],
        token_vocab=token_vocab,
        label_vocab=label_vocab,
        batch_size=4,
        eval_ratio=0.1
    )
    # testdataloader = load_test_dataloader(
    #     data_files=['datas/original/test.csv'],
    #     token_vocab=token_vocab,
    #     label_vocab=label_vocab,
    #     batch_size=8
    # )
    print(len(traindataloader), len(evaldataloader))
    print('=' * 100)

    for data in traindataloader:
        print(data)
        break
    print('=' * 100)
    # for data in testdataloader:
    #     print(data)
    #     break
