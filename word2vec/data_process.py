"""
原始数据为 红楼梦.txt
数据处理步骤
文本分词：将文本转换为单词
构建一个单词到id的映射表（构建一个完整的词典对象）
构建训练数据（与模型相关）窗口大小决定训练数据的长度
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

# 需要根据分词结果不断添加自定义词
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
    if not os.path.exists(os.path.dirname(out_file)):  # 输出文件的目录不存在
        os.makedirs(os.path.dirname(out_file))  # 创建输出文件的目录
    with open(in_file, 'r', encoding='utf-8') as reader, open(out_file, 'w', encoding='utf-8') as writer:
        for line in reader:
            line = line.strip()  # 去除首尾空白字符
            if not line:  # 去除空白行
                continue
            # 对每行文本分词, 分词结果以列表形式返回
            words = jieba.lcut(line)
            # 分词结果小于10的句子删除
            if len(words) < 10:
                continue
            # 将文本按空格串联输出为一行
            text = " ".join(words)
            writer.writelines(f'{text}\n')


class Vocab:
    def __init__(self, token2count, total_count, tokens=None):
        """
        构建词典映射关系
        :param token2count: 负采样时需要知道每个token的出现频率，字典key：token，value：count
        :param total_count: token的总数
        :param tokens: 默认不输入，如果存在为列表形式
        """
        self.tokens = tokens
        self.token_to_idx = dict()  # 输入token获取id，字典索引键获取值
        self.idx_to_token = list()  # 输入id获取token，列表索引实现
        self.token2count = token2count
        self.total_count = total_count

        if self.tokens is not None:  # 输入的token非空
            if '<UNK>' not in self.tokens:  # 将'<UNK>'添加到tokens列表中
                self.tokens = self.tokens + ['<UNK>']
            for token in self.tokens:
                self.idx_to_token.append(token)  # 将token储存在列表中
                self.token_to_idx[token] = len(self.idx_to_token) - 1  # 将token：id键值对储存在字典中，从0开始
            self.unk = self.token_to_idx['<UNK>']

    @classmethod
    def build_vocabulary(cls, in_file, min_freq=2, reversed_tokens=None):
        """
        类方法，构建词典对象
        in_file:分好词的文件路径
        min_freq:出现最低次数，低于此数，按unk处理
        reversed_tokens:保留的tokens，默认无。  如果次数低于min_freq，但是在reversed_tokens中，保留下来
        """
        token_freq = defaultdict(int)  # 字典，值为int类型
        # 解析输入文件，并获取每个单词的词频
        with open(in_file, 'r', encoding='utf-8') as reader:
            for line in reader:
                tokens = line.strip().split(' ')  # split成列表，每个元素为词
                for token in tokens:
                    token_freq[token] += 1  # 统计每个token的出现次数
        # 构造tokens列表  添加'<PAD>', '<UNK>'及 reversed_tokens
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
        """返回id对应的token"""
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
        """加载vocab.pkl文件"""
        with open(file, 'rb') as reader:
            obj = pickle.load(reader)
            # 输入Vocab所需的三个参数（从obj中获取）即可构建Vocab对象
            return Vocab(
                token2count=obj['token2count'],
                total_count=obj['total_count'],
                tokens=obj['tokens']
            )


def build_training_data(in_file, out_file, vocab: Vocab, window_size=5, start_first=False, pad_token='<PAD>'):
    """
    cbow: 窗口大小固定为5(1中心词 + 4周边词)
    start_first: 中心词从第一个位置开始，则需要向首尾两侧各加两个pad_token
    """
    if not os.path.exists(os.path.dirname(out_file)):
        os.mkdir(os.path.dirname(out_file))
    # creat_dir(os.path.dirname(out_file))

    with open(in_file, 'r', encoding='utf-8') as reader, open(out_file, 'w', encoding='utf-8') as writer:
        m1 = (window_size - 1) // 2
        m2 = window_size - 1 - m1
        for sentence in reader:
            sentence = sentence.strip()
            if start_first:
                v1 = ' '.join(map(lambda t: pad_token, range(m1)))
                v2 = ' '.join(map(lambda t: pad_token, range(m2)))
                sentence = f'{v1} {sentence} {v2}'   # 向首尾两侧各加两个pad_token
            words = sentence.split(' ')
            if len(words) < window_size:  # 去除长度小于窗口大小的句子
                continue
            # 将token转换为id，
            words = [str(vocab.token2idx(word)) for word in words]
            # 对每行文本数据按窗口滑动提取(x, y)
            for i in range(m1, len(words)-m2):
                pre_words = words[i-m1:i]  # 中心词前面m1个词
                next_words = words[i+1:i+1+m2]  # 中心词后面m1个词
                mid_word = words[i]  # 中心词
                # 将每个窗口的每个token的id写入writer文件中，每一行为一个窗口
                writer.writelines(f"{' '.join(pre_words)} {' '.join(next_words)} {mid_word}\n")

class Word2VecDataset(Dataset):
    def __init__(self, in_file):
        """
        in_file: 按窗口大小分好的数据， eg:0 0 2 3 1
        """
        self.records = []
        with open(in_file, 'r', encoding='utf-8') as reader:
            for sentence in reader:
                # 将每行数字转化为int
                words = list(map(int, sentence.strip().split(' ')))
                # 以(x, y)形式构造输入x和标签y
                self.records.append((words[0:-1], words[-1]))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index) -> T_co:
        context_words, central_words = self.records[index]
        return torch.tensor(context_words), torch.tensor(central_words)


if __name__ == '__main__':
    # 分词
    split_file_content(in_file='./datas/红楼梦.txt', out_file='./output/datas/红楼梦_words.txt')

    # 构建词典
    vocab = Vocab.build_vocabulary('./output/datas/红楼梦_words.txt', min_freq=2, reversed_tokens=None)
    vocab.save('./output/datas/vocab.pkl')
    vocab2 = Vocab.load('./output/datas/vocab.pkl')
    print(vocab.token2idx('贾宝玉'))
    print(vocab.idx_to_token[3])
    print(vocab2.idx_to_token[0])

    # 训练数据构建
    build_training_data(
        in_file='./output/datas/红楼梦_words.txt',
        out_file='./output/datas/红楼梦_training.txt',
        vocab=Vocab.load('./output/datas/vocab.pkl'),
        start_first=True
    )
    # 数据迭代器处理
    dataset = Word2VecDataset('./output/datas/红楼梦_training.txt')
    print(f"总样本数量:{len(dataset)}")
    print(dataset[546])

