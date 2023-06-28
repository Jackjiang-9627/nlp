"""
定义数据处理相关code
"""
import pickle
from collections import defaultdict
import os
import random

import numpy as np
import torch
import jieba
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

JIEBA_USER_DICT_PATH = os.path.abspath(os.path.join(__file__, '..', 'datas', 'jieba_dictionary.txt'))
jieba.load_userdict(JIEBA_USER_DICT_PATH)


def split_text(text):
    return jieba.lcut(text)


def creat_dir(dir_path):
    """创建目录"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def split_file_content(in_file, out_file):
    """
    对输入文件中的所有文本进行分词，并将结果保存到输出文件中
    :param in_file: 输入文件路径
    :param out_file:输出文件路径
    :return: 一个分词文件
    """
    if not os.path.exists(in_file):
        raise ValueError(f'输入文件{in_file}不存在')
    creat_dir(os.path.dirname(out_file))
    with open(in_file, 'r', encoding='utf-8') as reader, open(out_file, 'w', encoding='utf-8') as writer:
        for line in reader:
            line = line.strip()
            if not line:  # 空行去掉
                continue
            # 文本分词
            words = split_text(line)
            # 分词结果小于10的句子删除
            if len(words) < 10:
                continue
            # 将文本按空格串联输出
            text = " ".join(words)
            writer.writelines(f'{text}\n')


class Vocab:
    def __init__(self, token2count, total_count, tokens=None):
        self.tokens = tokens
        self.token_to_idx = dict()
        self.idx_to_token = list()
        self.token2count = token2count
        self.total_count = total_count

        if self.tokens is not None:
            if '<UNK>' not in self.tokens:
                self.tokens = self.tokens + ['<UNK>']
            for token in self.tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
            self.unk = self.token_to_idx['<UNK>']

    @classmethod
    def build_vocabulary(cls, in_file, min_freq=2, reversed_tokens=None):

        token_freq = defaultdict(int)
        # 解析输入文件，并获取每个单词的词频
        with open(in_file, 'r', encoding='utf-8') as reader:
            for line in reader:
                tokens = line.strip().split(' ')
                for token in tokens:
                    token_freq[token] += 1
        uniq_tokens = ['<PAD>', '<UNK>'] + (reversed_tokens if reversed_tokens else [])
        # 去除低词频单词并重新统计词频
        token2count = defaultdict(int)
        total_count = 0
        for token, freq in token_freq.items():
            if freq > min_freq and token not in ['<PAD>', '<UNK>']:
                uniq_tokens += [token]
                token2count[token] = freq
                total_count += freq

        return cls(token2count, total_count, uniq_tokens)

    def __len__(self):
        return len(self.idx_to_token)

    def token2idx(self, token):
        """返回token的id, 不存在返回<UNK>的id == 1"""
        return self.token_to_idx.get(token, self.unk)

    def idx2token(self, idx):
        return self.idx_to_token[idx]

    def fetch_negative_sampling_token_frequency(self, power_coff=0.75):
        """
        获取负采样过程中的token频率数组
        :param power_coff: 幂次方的次数， 默认0.75
        :return: 1-D数组，按照token顺序给定的频率值
        """
        total = 0
        counts = []
        # 遍历
        for idx in range(len(self.idx_to_token)):
            cnt = self.token2count.get(self.idx_to_token[idx], 0)
            cnt = np.power(cnt, power_coff)
            counts.append(cnt)
            total += cnt
        return [1 * cnt / total for cnt in counts]

    def save(self, file):
        creat_dir(os.path.dirname(file))
        with open(file, 'wb') as writer:
            obj = {
                'token2idx': self.token2idx,
                'idx2token': self.idx2token,
                'token2count': self.token2count,
                'total_count': self.total_count,
                'tokens': self.tokens
            }
            pickle.dump(obj, writer)

    @staticmethod
    def load(file):
        with open(file, 'rb') as reader:
            obj = pickle.load(reader)
            return Vocab(
                token2count=obj['token2count'],
                total_count=obj['total_count'],
                tokens=obj['tokens']
            )
def build_training_data(in_file, out_file, vocab:Vocab, window_size=5, start_first=False, pad_token='<PAD>'):
    creat_dir(os.path.dirname(out_file))
    with open(in_file, 'r', encoding='utf-8') as reader, open(out_file, 'w', encoding='utf-8') as writer:
        m1 = (window_size - 1) // 2
        m2 = window_size -1 - m1
        for sentence in reader:
            sentence = sentence.strip()
            if start_first:
                v1 = ' '.join(map(lambda t:pad_token, range(m1)))
                v2 = ' '.join(map(lambda t:pad_token, range(m2)))
                sentence = f'{v1} {sentence} {v2}'
            words = sentence.split(' ')
            if len(words) < window_size:
                continue
            # 将token转换为token的id
            words = [str(vocab.token2idx(word)) for word in words]
            # 对每行文本数据按窗口滑动提取(x, y)
            for i in range(m1, len(words)-m2):
                pre_words = words[i-m1:i]
                next_words = words[i+1:i+1+m2]
                mid_word = words[i]
                writer.writelines(f"{' '.join(pre_words)} {' '.join(next_words)} {mid_word}\n")

class Word2VecDataset(Dataset):
    def __init__(self, in_file):
        # 解析文件
        self.records = []
        with open(in_file, 'r', encoding='utf-8') as reader:
            for sentence in reader:
                words = list(map(int, sentence.strip().split(' ')))
                self.records.append((words[0:-1], words[-1]))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index) -> T_co:
        context_words, central_words = self.records[index]
        return torch.tensor(context_words), torch.tensor(central_words)


if __name__ == '__main__':
    # 分词
    # split_file_content(in_file='./datas/人民的名义.txt', out_file='./output/datas/人民的名义_words.txt')

    # 构建词典
    # vocab = Vocab.build_vocabulary('./output/datas/人民的名义_words.txt', min_freq=2, reversed_tokens=None)
    # vocab.save('./output/datas/vocab.pkl')
    # vocab2 = Vocab.load('./output/datas/vocab.pkl')
    # print(vocab.token2idx('侯亮平'))
    # print(vocab2.token2idx('侯亮平'))
    # print(vocab.idx_to_token[3])
    # print(vocab2.idx_to_token[0])

    # 训练数据构建
    # build_training_data(
    #     in_file='./output/datas/人民的名义_words.txt',
    #     out_file='./output/datas/人民的名义_training.txt',
    #     vocab=Vocab.load('./output/datas/vocab.pkl'),
    #     start_first=True
    # )
    # 数据迭代器处理
    dataset = Word2VecDataset('./output/datas/人民的名义_training.txt')
    print(f"总样本数量:{len(dataset)}")
    print(dataset[546])

