import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from models.encoder_models import EncoderModule
from utils import Params


def clones(model, n):
    return nn.ModuleList([copy.deepcopy(model) for _ in range(n)])

class PositionEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, dropout, max_len=1000):
        """
        :param d_model: token维度
        :param max_len: 给定足够长的位置
        :param dropout: 概率值
        """
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 初始化位置嵌入矩阵PE:[N=1, T, d_model]
        self.PE = torch.zeros((1, max_len, d_model))
        # x = i / 10000^(2j/d)  i:T维度， j:E维度
        # X = i / e^(2j * ln(10000)/d) = i * e^-(2j * ln(10000)/d) = i * e^(2j * -ln(10000)/d)
        position = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        X = position * div_term
        # [max_len, 1] * [d_model / 2, ] --> [max_len, 1] * [1, d_model / 2]
        # --> [max_len, d_model/2] * [max_len, d_model/2] --> [max_len, d_model/2]
        # self.PE[..., 0::2] : [1, 1000, 16]
        self.PE[..., 0::2] = torch.sin(X)
        self.PE[..., 1::2] = torch.cos(X)

    def forward(self, X):
        """
        词向量[N, T, E]与位置编码[N, max_len=1000 > T, E]相加时需要对PE切片取前T个位置
        :param X: [N, max_len, d_model]
        """
        X = X + self.PE[:, :X.shape[1], :].to(X.device)
        return X


class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, x, sublayer):
        return self.ln(self.dropout(sublayer) + x)

def attention(q, k, v, mask=None, dropout=None):
    """
    计算缩放点积注意力
    :param q:[N, num_heads, T, head_dim]
    :param k:[N, num_heads, T, head_dim]
    :param v:[N, num_heads, T, head_dim]
    :param mask: None 或 [1, 1, T, T]
    :param dropout: None 或 dropout
    :return:加权后的 Value[N, num_heads, T, head_dim] 和 attention scores [N, num_heads, T, head_dim]
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])  # [N, num_heads, T, T]
    # decoder: auto-regression
    if mask is not None:  # mask操作，mask取非常小的负数，e的指数趋于0
        scores = scores.masked_fill(mask.to('cuda') == 0, -1e9)  # mask数据to gpu， 否则报错
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    V = torch.matmul(p_attn, v)
    return V, p_attn

class MHA(nn.Module):
    """
    缩放点积注意力
    多头注意力融合了来自多个自注意力汇聚的不同知识，这些知识的不同来源于相同的查询、键和值的不同的表示子空间
    适当的张量操作，可以实现多头注意力的并行计算
    """
    def __init__(self, num_heads, d_model, dropout):
        super(MHA, self).__init__()
        assert d_model % num_heads == 0, f'd_model={d_model}不是num_heads={num_heads}的整数倍'
        self.head_dim = d_model // num_heads
        self.num_heads = num_heads
        # Q、K、V的转换以及heads concat后的linear，一共4个linear
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

        # auto-regressive: 解码时用
        attn_shape = (1, 3000, 3000)  # 3000为T维度大小，可以换成最大序列长度
        # 上三角为1
        attn_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        # 下三角为1，第1个token只包含位置1的信息，第2个包含位置1、2的信息，....
        # [1, 1, 3000, 3000]为了与Value:[N, num_heads, T, head_dim]相乘
        self.mask = (torch.from_numpy(attn_mask) == 0).unsqueeze(1)

    def forward(self, x):
        """
        :param x: [N, T, d_model]
        :return:
        """
        batch_size, seq_len, d_model = x.shape
        # 1、线性变换获取qkv
        # 为了多注意力的并行计算需要变换qkv的形状
        # 由[N, T, d_model] --> [N, T, num_heads, head_dim] --> [N, num_heads, T, head_dim]
        q, k, v = [liner(xx).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                   for liner, xx in zip(self.linears, (x, x, x))]
        # 2、attention得到：加权后的Value和attention分数
        # Value:[N, num_heads, T, head_dim]
        # attention:[T, T]
        x, self.attn = attention(q, k, v, mask=self.mask[:, :, :seq_len, :seq_len], dropout=self.dropout)

        # 3、final linear 拼接concat 多头注意力结果
        # tensor对象transpose后必须接contiguous才能使用view
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        return self.linears[-1](x)


class PositionWiseFFN(nn.Module):
    def __init__(self, input_size, hiddens_size, output_size):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(input_size, hiddens_size)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(hiddens_size, output_size)

    def forward(self, x):
        return self.dense2(self.relu(self.dense1(x)))


class EncoderBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, num_heads, d_model, dropout, norm_shape):
        super(EncoderBlock, self).__init__()
        self.attention = MHA(num_heads, d_model, dropout)
        self.connections = clones(AddNorm(norm_shape, dropout), 2)
        self.ffn = PositionWiseFFN(
            input_size=d_model,
            hiddens_size=d_model * num_heads,
            output_size=d_model
        )

    def forward(self, x):
        # multi-head attention
        x = self.connections[0](x, self.attention(x))
        # fnn
        x = self.connections[1](x, self.ffn(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_blocks, num_heads, d_model, dropout, norm_shape):
        super(TransformerEncoder, self).__init__()
        # pos_encoding输入
        self.pos_encoding = PositionEncoding(d_model, dropout)
        self.blocks = nn.Sequential()
        for i in range(num_blocks):
            self.blocks.add_module(
                name='block'+str(i),
                module=EncoderBlock(num_heads, d_model, dropout, norm_shape)
            )

    def forward(self, x):
        # 因为位置编码范围在-1~1之间，所以需要检查嵌入值是否也是在-1~1之间，若不在进行缩放，再与位置编码相加
        x = self.pos_encoding(x)
        x = self.blocks(x)
        return x


class DecoderBlock(nn.Module):
    pass


class TransformerDecoder(nn.Module):
    pass


class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, head_dims):
        super(Transformer, self).__init__()
        self.Encoder = TransformerEncoder
        self.Decoder = TransformerDecoder
        self.linear = nn.Linear(d_model, num_heads * head_dims)

    def forward(self, x):

        return x


class TransformerEncoderModule(EncoderModule):
    def __init__(self, params: Params):
        super(TransformerEncoderModule, self).__init__(params)
        self.proxy = Transformer()

    def forward(self, input_feature, input_mask, **kwargs):
        return self.proxy(input_feature, input_mask)


