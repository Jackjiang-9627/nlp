import torch
from torch.utils.data.dataset import T_co
from torch.nn.utils.rnn import pad_sequence
from .vocab import Vocab
from torch.utils.data import DataLoader, Dataset
from nltk.corpus import sentence_polarity


def load_sentence_polarity():
    # 使用全部句子集合（已经过标记解析）创建词表
    vocab = Vocab.build(sentence_polarity.sents())
    train_data_pos = [(vocab.convert_tokens_to_ids(sentence), 0)
                      for sentence in sentence_polarity.sents(categories='pos')[:4000]]
    train_data_neg = [(vocab.convert_tokens_to_ids(sentence), 1)
                      for sentence in sentence_polarity.sents(categories='neg')[:4000]]
    train_data = train_data_pos + train_data_neg
    test_data_pos = [(vocab.convert_tokens_to_ids(sentence), 0)
                     for sentence in sentence_polarity.sents(categories='pos')[4000:]]
    test_data_neg = [(vocab.convert_tokens_to_ids(sentence), 1)
                     for sentence in sentence_polarity.sents(categories='neg')[4000:]]
    test_data = test_data_pos + test_data_neg
    return train_data, test_data, vocab

def length_to_mask(lengths):
    """将序列长度转换为mask矩阵"""
    max_len = torch.max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(lengths.shape[0], max_len) < lengths.unsqueeze(1)
    return mask

class BowDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> T_co:
        return self.data[index]


def collate_fn_mlp(examples):
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    offsets = [0] + [i.shape[0] for i in inputs]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    inputs = torch.cat(inputs)
    return inputs, offsets, targets

def collate_fn_cnn(examples):
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    # 对批次内的样本进行补齐，使其具有相同长度
    inputs = pad_sequence(inputs, batch_first=True)
    return inputs, targets

def collate_fn_lstm(examples):
    # 获取每个序列长度
    lengths = torch.tensor([len(ex[0]) for ex in examples])
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    # 对批次内的样本进行补齐，使其具有相同长度
    inputs = pad_sequence(inputs, batch_first=True)
    return inputs, lengths, targets
