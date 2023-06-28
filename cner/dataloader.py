import os.path
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.dataset import T_co
from transformers import BertTokenizer, BertConfig


from utils import Params, read_examples, examples2features, InputFeature


class FeatureDataset(Dataset):
    def __init__(self, features: list):
        """
        :param features: 文本转化为数值后的数据 [InputFeature]
        self.input_ids = input_ids
        self.label_ids = label_ids
        self.input_mask = input_mask
        """
        super(FeatureDataset, self).__init__()  # 继承父类构造函数中的内容，且子类需要在父类的基础上补充
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> T_co:
        return self.features[i]


class NERDataLoader(object):
    def __init__(self, params: Params):
        super(NERDataLoader, self).__init__()
        self.tag2idx = params.tag2idx
        self.data_cache = params.data_cache
        self.max_seq_length = params.max_seq_len
        self.tokenizer = BertTokenizer(params.bert_vocab_path, do_lower_case=True)
        self.data_dir = params.data_dir  # 读取datas文件下的数据
        self.bert_root_dir = params.bert_root_dir  # 将缓存数据保存到pre_train_models文件夹下的各个模型文件中
        self.test_batch_size = params.test_batch_size
        self.val_batch_size = params.val_batch_size
        self.train_batch_size = params.train_batch_size

    def get_features(self, data_sign):
        # 如果存在features缓存文件，直接导入
        cache_path = os.path.join(self.bert_root_dir, f'{data_sign}.cache.{self.max_seq_length}')
        if os.path.exists(cache_path) and self.data_cache:
            print(f'直接加载{data_sign}对应的缓存数据集{cache_path}')
            features = torch.load(cache_path, map_location='cpu')
        else:
            # 1、加载数据：InputExample
            print('=**=' * 10)
            print(f'加载{data_sign}的InputExample类型数据集...')
            if data_sign in ['train', 'test', 'val']:
                examples = read_examples(self.data_dir, data_sign)
            else:
                raise ValueError(f'数据类型参数异常：{data_sign}，仅支持train、val、test')
            # 2、将InputExample数据转换为InputFeature数据
            features = examples2features(
                examples=examples,
                tokenizer=self.tokenizer,
                word_dict={},
                tag2idx=self.tag2idx,
                max_seq_len=self.max_seq_length,
                greedy=False,
                pad_sign=True,
                pad_token='[PAD]'
            )
            if self.data_cache:
                torch.save(features, cache_path)
        return features

    @staticmethod
    def collate_fn(batch: List[InputFeature]):
        """对一个批次进行数据处理成需要的结构，batch为InputFeature的列表"""
        input_ids = torch.tensor([f.input_ids for f in batch], dtype=torch.long)
        label_ids = torch.tensor([f.label_ids for f in batch], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in batch], dtype=torch.long)
        tensor = [input_ids, input_mask, label_ids]
        return tensor

    def get_dataloader(self, data_sign='train'):
        """
        获取PyTorch模型需要的DataLoader对象
        :param data_sign:可选值：train、val、test
        :return:
        """
        # 获取特征对象
        features = self.get_features(data_sign)
        dataset = FeatureDataset(features=features)
        print(f'{len(features)} {data_sign}数据集加载完成!')
        print('=' * 50)

        if data_sign == 'train':
            batch_size = self.train_batch_size
            data_sampler = RandomSampler(dataset)
        elif data_sign == 'val':
            batch_size = self.val_batch_size
            data_sampler = SequentialSampler(dataset)
        elif data_sign == 'test':
            batch_size = self.test_batch_size
            data_sampler = SequentialSampler(dataset)
        else:
            raise ValueError(f'数据类型{data_sign}异常，仅支持：train、val、test')
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            sampler=data_sampler  # 有这个参数就不要shuffle
        )
        return dataloader


if __name__ == '__main__':
    # cfg = None  # 直接导入BertConfig对象的参数
    cfg = BertConfig.from_json_file(r"./pre_train_models/bert/config.json")
    params = Params(
        config=cfg,
        params={
            'data_dir': r'.\datas\sentence_tag',
            'bert_root_dir':  r'./pre_train_models/bert'
        }
    )
    train_dataloader = NERDataLoader(params).get_dataloader('train')
    i = 0
    for input_ids, input_mask, label_ids in train_dataloader:
        i += 1
    print(i)
