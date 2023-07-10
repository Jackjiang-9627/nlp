import torch.nn as nn
import sys
sys.path.append('./models')
from classify_model import MLPModule
from embedding import TokenEmbeddingModule
from seq_feature_extract_model import *


class TextClassifyModel(nn.Module):
    def __init__(self, embedding_layer: TokenEmbeddingModule,
                 seq_feature_extract_layer,
                 classify_decision_layer: MLPModule
                 ):
        """
        构建函数
        :param embedding_layer: 获取序列中每个token的对应向量
        :param seq_feature_extract_layer: 获取整个序列/样本的对应特征向量
        :param classify_decision_layer: 基于最终的序列/样本特征向量进行决策输出
        """
        super(TextClassifyModel, self).__init__()
        self.embedding_layer = embedding_layer
        self.seq_feature_extract_layer = seq_feature_extract_layer
        self.classify_decision_layer = classify_decision_layer

    def forward(self, input_ids, seq_lengths):
        input_ids = input_ids.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        seq_lengths = seq_lengths.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # 1. 获取token的embedding向量     [N,T] -> [N,T,E1]
        embedding = self.embedding_layer(input_ids)
        # 2. 获取序列的特征向量            [N,T,E1] -> [N,E2]
        seq_feature = self.seq_feature_extract_layer(embedding, seq_lengths)
        # 3. 决策输出                     [N,E2] -> [N,num_classes]
        score = self.classify_decision_layer(seq_feature)
        return score

    @staticmethod
    def build_model(cfg, weights=None, strict=False):
        model = TextClassifyModel.parse_model(cfg)
        if weights is not None:
            print("进行模型恢复!!")
            if isinstance(weights, str):  # weights为模型路径，转变为静态
                weights = torch.load(weights, map_location='cpu').state_dict()
            elif isinstance(weights, TextClassifyModel):
                weights = weights.state_dict()
            # missing_keys: model当前需要参数，但是weights这个dict中没有给定 --> 也就是没有恢复的参数
            # unexpected_keys: weights中有，但是model不需要的参数
            missing_keys, unexpected_keys = model.load_state_dict(weights, strict=False)
            if strict and (len(missing_keys) > 0):
                raise ValueError(f"模型存在部分参数未恢复的情况:{missing_keys}")
        return model

    @staticmethod
    def parse_model(cfg):
        # 构建token向量提取层对象
        model = eval(cfg['token_emb_layer']['name'])
        args = cfg['token_emb_layer']['args']  # 模型所需参数
        token_emb_layer = model(*args)
        # 构建序列特征向量提取层对象
        model = eval(cfg['seq_feature_extract_layer']['name'])
        args = cfg['seq_feature_extract_layer']['args']
        # args:List      [256, 'all_mean', 2, True]
        # 将上一层的输出向量大小作为当前层的输入   [128, 256, 'all_mean', 2, True]
        args.insert(0, token_emb_layer.output_size)
        seq_feature_extract_layer = model(*args)
        # 构建决策输出层对象
        model = eval(cfg['classify_decision_layer']['name'])
        args = cfg['classify_decision_layer']['args']
        args.insert(0, seq_feature_extract_layer.hidden_size)  # 将上一层的输出向量大小作为当前层的输入
        classify_decision_layer = model(*args)
        return TextClassifyModel(token_emb_layer, seq_feature_extract_layer, classify_decision_layer)
