import os
import re
import torch
import jieba
from torch import nn

from models.model import TextClassifyModel

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))


def get_project_absolute_path(path):
    """返回项目的绝对路径"""
    return os.path.abspath(os.path.join(ROOT_DIR, path))


jieba.load_userdict(get_project_absolute_path('./datas/jieba_words.txt'))  # 注意：此处应该用绝对路径！
# jieba.suggest_freq(['xx', 'xx'], True)
re_number = re.compile("([0-9\.]+)", re.U)
re_punctuation = re.compile("([,.，。？！：；、“\"]+)", re.U)


def split_text_token(text):
    return jieba.lcut(text)


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


class Predictor(object):
    """要求：恢复模型、针对给定的文本返回对应的预测类别以及概率值、支持topk返回"""
    from models.model import TextClassifyModel

    def __init__(self, ckpt_path, token_vocab_file, label_vocab_file, num_token='<NUM>', pun_token='<PUN>'):
        # 1. 映射字典恢复
        self.token_vocab = torch.load(token_vocab_file, map_location='cpu')
        self.label_vocab = torch.load(label_vocab_file, map_location='cpu')
        self.num_token = num_token
        self.pun_token = pun_token
        # 2. 模型恢复
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        weights = None
        cfg = {
            'token_emb_layer': {
                'name': 'TokenEmbeddingModule',
                'args': [len(self.token_vocab), 128]
            },
            'seq_feature_extract_layer': {
                'name': 'LSTMSeqFeatureExtractModule',  # 注意：此处的args参数顺序与模型定义的参数顺序有关
                'args': [256, 'all_mean', 2, True]
            },
            'classify_decision_layer': {
                'name': 'MLPModule',
                'args': [len(self.label_vocab), [256, 128], 0.1, nn.ReLU(), True]
            }
        }
        model = TextClassifyModel.build_model(cfg, weights)
        model.load_state_dict(ckpt['model'])
        self.model = model
        self.model.eval()  # 进行推理阶段
        self.model.to(self.device)



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
