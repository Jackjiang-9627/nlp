"""R-Transformer layer"""

import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from models.encoder_models import EncoderModule

def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

class LayerNorm(nn.Module):
    def __init__(self, feature_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(feature_size), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(feature_size), requires_grad=True)

    def forward(self, x):  # todo: 检验是否在token向量维度上取均值、方差， alpha与x之间的运算过程
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta

class SublayerConnection(nn.Module):
    """
    有三个(sublayer)子层:RNN、MHA、FFN
    在每个子层之间做一次 add + norm
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.ln = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.ln(x)))


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation.两个全连接，中间隐藏层大小为 4*d_model"""
    def __init__(self, d_model, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        d_ff = d_model * 4
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

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
    # scores = torch.matmul(q, k.transpose(-2, -1))
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])  # [N, num_heads, T, T]
    # decoder: auto-regression
    if mask is not None:  # mask操作，mask取非常小的负数，e的指数趋于0
        scores = scores.masked_fill(mask.to('cuda') == 0, -1e9)  # mask数据to gpu， 否则报错
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    V = torch.matmul(p_attn, v)
    return V, p_attn

class MHPooling(nn.Module):
    """
    缩放点积注意力
    多头注意力融合了来自多个自注意力汇聚的不同知识，这些知识的不同来源于相同的查询、键和值的不同的表示子空间
    适当的张量操作，可以实现多头注意力的并行计算
    """
    def __init__(self, d_model, num_heads, dropout):
        super(MHPooling, self).__init__()
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
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        # 下三角为1，第1个token只包含位置1的信息，第2个包含位置1、2的信息，....
        # [1, 1, 3000, 3000]为了与Value:[N, num_heads, T, head_dim]相乘
        self.mask = (torch.from_numpy(subsequent_mask) == 0).unsqueeze(1)

    def forward(self, x):
        """
        :param x: [N, T, d_model]
        :return:
        """
        batch_size, seq_len, d_model = x.shape
        # 1、线性变换获取qkv
        # 为了多注意力的并行计算需要变换qkv的形状
        # 由[N, T, d_model] --> [N, T, num_heads, head_dim] --> [N, num_heads, T, head_dim]
        q, k, v = [linear(xx).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                   for linear, xx, in zip(self.linears, (x, x, x))]

        # 2、attention得到：加权后的Value和attention分数
        # Value:[N, num_heads, T, head_dim]
        # attention:[N, num_heads, T, head_dim]
        x, self.attn = attention(q, k, v, mask=self.mask[:, :, :seq_len, :seq_len],
                                 dropout=self.dropout)

        # 3、final linear 拼接concat 多头注意力结果
        # tensor对象transpose后必须接contiguous才能使用view
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        return self.linears[-1](x)


class LocalRNN(nn.Module):
    def __init__(self, input_dim, output_dim, rnn_type, ksize):
        """
        input_dim = output_dim
        :param input_dim:
        :param output_dim:
        :param rnn_type:
        :param ksize: rnn序列滑动窗口的长度
        """
        super(LocalRNN, self).__init__()
        self.ksize = ksize
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(input_dim, output_dim)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_dim, output_dim)
        else:
            self.rnn = nn.RNN(input_dim, output_dim)
        # To speed up 按照窗口滑动构造序列id组 ksize=3
        # [0,...,k_size-1, 1,...,k_size, 2,...,k_size+1 ... len-k_size,...,len-1]
        # [0, 1, 2] + [1,2, 3] + [2, 3, 4] + ...
        idx = [i for j in range(ksize-1, 10000) for i in range(j-(ksize-1), j+1, 1)]
        self.idx = torch.tensor(idx, dtype=torch.long)
        self.zeros = torch.zeros((self.ksize-1, input_dim))

    def get_k(self, x):
        """将输入加滑动窗口
        :param x: (batch_size, max_len, input_dim)
        :return: key: split to kernel size. (batch_size, l, ksize, input_dim)
        """
        batch_size, l, d_model = x.shape
        zeros = self.zeros.unsqueeze(0).repeat(batch_size, 1, 1).to(x.device)  # (batch_size, ksize-1, input_dim=d_model)
        # 在序列首端增加两个0
        x = torch.cat((zeros, x), dim=1)
        key = torch.index_select(x, 1, self.idx[:self.ksize * l].to(x.device))  # (batch_size, ksize*l, input_dim)
        # 将输入按rnn窗口大小构造出key --> (batch_size, l, ksize, input_dim)
        key = key.reshape(batch_size, l, self.ksize, -1)
        return key

    def forward(self, x):  # x:[batch_size, max_len, input_dim]
        # 将输入数据转换为按窗口重新构造的格式 --> [batch_size, l, ksize, input_dim]
        x = self.get_k(x)
        b, l, ksize, d_model = x.shape
        # input: [batch_size*max_len, ksize, d_model]
        h, _ = self.rnn(x.view(-1, ksize, d_model))
        # 窗口滑动操作使每个token重复了ksize次，前两次由于前面补的两个0，将会遗漏实际的第一个token信息
        h = h[:, -1, :]
        # output: [batch_size, max_len, 1, d_model]
        return h.view(b, l, d_model)

class LocalRNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, rnn_type, ksize, dropout):
        super(LocalRNNLayer, self).__init__()
        self.local_rnn = LocalRNN(input_dim, output_dim, rnn_type, ksize)
        self.connection = SublayerConnection(output_dim, dropout)

    def forward(self, x):
        z = self.connection(x, self.local_rnn)
        return z

class Block(nn.Module):
    def __init__(self, num_rnnlayers, input_dim, output_dim, rnn_type, ksize, dropout, num_heads):
        super(Block, self).__init__()
        # rnn层
        self.rnn_layers = clones(
            LocalRNNLayer(input_dim, output_dim, rnn_type, ksize, dropout), num_rnnlayers)
        # multi-head attention
        self.pooling = MHPooling(input_dim, num_heads, dropout)
        # FFN全连接层
        self.feed_forward = PositionwiseFeedForward(input_dim, dropout)
        # 每层之间做一个add（残差）+ LN连接, rnn层与multi-head attention之间、 multi-head attention与FFN全连接层之间
        self.connections = clones(SublayerConnection(output_dim, dropout), 2)

    def forward(self, x):
        # n, l, d = x.shape
        # local rnn:短序列提取特征，从而保持特征中具有方向以及距离等相关信息
        for i, layer in enumerate(self.rnn_layers):
            x = layer(x)
        # multi-head attention
        x = self.connections[0](x, self.pooling)
        # fnn
        x = self.connections[1](x, self.feed_forward)
        return x

class RTransformer(nn.Module):
    def __init__(self, d_model, encoder_output_size, num_block, num_rnnlayers,
                rnn_type, ksize, dropout, num_heads):
        super(RTransformer, self).__init__()
        layers = [
            Block(num_rnnlayers, d_model, d_model, rnn_type, ksize, dropout, num_heads)
            for _ in range(num_block)
        ]

        self.forward_net = nn.Sequential(*layers)
        self.output_layer = nn.Linear(d_model, encoder_output_size)

    def forward(self, x, mask):
        z = self.forward_net(x)
        z = z * mask.unsqueeze(-1)
        z = self.output_layer(z) * mask.unsqueeze(-1)
        return z


class RTransformerEncoderModule(EncoderModule):
    """
    p = 0.3, lr = 1e-4  --> 0.827, 0.690, 1.140, 78.160, 0.962
    p = 0.3, lr = 1e-3  --> 0.994, 0.711, 1.118, 77.071, 0.962
    """
    def __init__(self, params):
        super(RTransformerEncoderModule, self).__init__(params)
        self.proxy = RTransformer(
            d_model=self.params.config.hidden_size,
            encoder_output_size=self.params.encoder_output_size,
            num_block=1,
            num_rnnlayers=1,
            rnn_type='GRU',
            ksize=10,
            dropout=self.params.encoder_dropout,
            num_heads=4
        )

    def forward(self, input_feature, input_mask, **kwargs):
        return self.proxy(input_feature, input_mask)


#
# import copy
#
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# from models.encoder_models import EncoderModule
#
#
# def clones(module, N):
#     """
#     Produce N identical layers. 模块复制
#     :param module: 待复制的模块
#     :param N: 复制N次
#     :return:
#     """
#     return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
#
#
# class LayerNorm(nn.Module):
#     def __init__(self, features, eps=1e-6):
#         """
#         Construct a LayerNorm module.
#         :param features: 312
#         :param eps: 非常小的数，防止分母为0
#         """
#         super(LayerNorm, self).__init__()
#         # a_1 a_2 需要学习的参数
#         self.a_2 = nn.Parameter(torch.ones(features), requires_grad=True)
#         self.b_2 = nn.Parameter(torch.zeros(features), requires_grad=True)
#         self.eps = eps
#
#     def forward(self, x):
#         mean = x.mean(-1, keepdim=True)
#         std = x.std(-1, keepdim=True)
#         return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
#
#
# class SublayerConnection(nn.Module):
#     """
#     A residual connection followed by a layer norm.
#     Note for code simplicity the norm is first as opposed to last.
#     """
#
#     def __init__(self, size, dropout):
#         super(SublayerConnection, self).__init__()
#         self.norm = LayerNorm(size)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x, sublayer):
#         """Apply residual connection to any sublayer with the same size."""
#         return x + self.dropout(sublayer(self.norm(x)))
#
#
# class PositionwiseFeedForward(nn.Module):
#     """Implements FFN equation."""
#
#     def __init__(self, d_model, dropout=0.1):
#         super(PositionwiseFeedForward, self).__init__()
#         d_ff = d_model * 4
#         self.w_1 = nn.Linear(d_model, d_ff)
#         self.w_2 = nn.Linear(d_ff, d_model)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         return self.w_2(self.dropout(F.relu(self.w_1(x))))
#
#
# def attention(query, key, value, mask=None, dropout=None):
#     """
#         Compute 'Scaled Dot Product Attention'
#         query, key, value : batch_size, n_head, seq_len, dim of space
#         :return output[0]: attention output. (batch_size, n_head, seq_len, head_dim)
#         :return output[1]: attention score. (batch_size, n_head, seq_len, seq_len)
#     """
#
#     # scores: batch_size, n_head, seq_len, seq_len
#     # scores = torch.matmul(query, key.transpose(-2, -1)) \
#     #          / math.sqrt(d_k)
#     scores = torch.matmul(query, key.transpose(-2, -1))
#
#     # auto-regression
#     if mask is not None:
#         scores = scores.masked_fill(mask.to('cuda') == 0, -1e9)  # mask数据to gpu， 否则报错
#
#     p_attn = F.softmax(scores, dim=-1)
#     if dropout is not None:
#         p_attn = dropout(p_attn)
#     return torch.matmul(p_attn, value), p_attn
#
#
# class MHPooling(nn.Module):
#     def __init__(self, d_model, h, dropout=0.1):
#         "Take in model size and number of heads."
#         super(MHPooling, self).__init__()
#         assert d_model % h == 0
#         # d_model = num_heads * head_dim
#         # We assume d_v always equals d_k
#         self.d_k = d_model // h
#         self.h = h
#         self.linears = clones(nn.Linear(d_model, d_model), 4)  # Q\K\V的转换 + 输出结果转换
#         self.attn = None
#         self.dropout = nn.Dropout(p=dropout)
#
#         # auto-regressive
#         attn_shape = (1, 3000, 3000)
#         subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
#         self.mask = (torch.from_numpy(subsequent_mask) == 0).unsqueeze(1)
#
#     def forward(self, x):
#         """Implements Figure 2
#         :param x: (batch_size, max_len, input_dim=d_model=output_dim)
#         :return:
#         """
#
#         nbatches, seq_len, d_model = x.shape
#
#         # 1) Do all the linear projections in batch from d_model => h x d_k
#         query, key, value = \
#             [l(xx).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
#              for l, xx in zip(self.linears, (x, x, x))]  # (batch_size, num_heads, max_len, head_dim)
#
#         # 2) Apply attention on all the projected vectors in batch.
#         # output(batch_size, n_head, seq_len, head_dim), attention score
#         x, self.attn = attention(query, key, value, mask=self.mask[:, :, :seq_len, :seq_len],
#                                  dropout=self.dropout)
#
#         # 3) "Concat" using a view and apply a final linear.
#         # (batch_size, seq_len, d_model)
#         x = x.transpose(1, 2).contiguous() \
#             .view(nbatches, -1, self.h * self.d_k)
#         return self.linears[-1](x)
#
#
# class LocalRNN(nn.Module):
#     def __init__(self, input_dim, output_dim, rnn_type, ksize):
#         super(LocalRNN, self).__init__()
#         """
#         LocalRNN structure
#         input_dim = output_dim
#         """
#         self.ksize = ksize
#         if rnn_type == 'GRU':
#             self.rnn = nn.GRU(input_dim, output_dim, batch_first=True)
#         elif rnn_type == 'LSTM':
#             self.rnn = nn.LSTM(input_dim, output_dim, batch_first=True)
#         else:
#             self.rnn = nn.RNN(input_dim, output_dim, batch_first=True)
#
#         # self.output = nn.Sequential(nn.Linear(output_dim, output_dim), nn.ReLU())
#
#         # To speed up
#         # [0,...,k_size-1, 1,...,k_size, 2,...,k_size+1 ... len-k_size,...,len-1]
#         idx = [i for j in range(self.ksize - 1, 10000, 1) for i in range(j - (self.ksize - 1), j + 1, 1)]
#         self.select_index = torch.tensor(idx, dtype=torch.long)
#         self.zeros = torch.zeros((self.ksize - 1, input_dim))  # (ksize-1, input_dim)
#
#     def forward(self, x):
#         """
#         :param x: (batch_size, max_len, input_dim)
#         :return: h: (batch_size, max_len, input_dim=d_model=output_dim)
#         """
#         x = self.get_K(x)  # b x seq_len x ksize x d_model
#         batch, l, ksize, d_model = x.shape
#         # input: (batch_size*max_len, ksize, d_model)
#         # output: (batch_size*max_len, 1, d_model)
#         h = self.rnn(x.view(-1, self.ksize, d_model))[0][:, -1, :]
#         return h.view(batch, l, d_model)
#
#     def get_K(self, x):
#         """将输入加滑动窗口
#         :param x: (batch_size, max_len, input_dim)
#         :return: key: split to kernel size. (batch_size, l, ksize, input_dim)
#         """
#         batch_size, l, d_model = x.shape
#         # zeros数据 --> 'gpu', 否则报错
#         zeros = self.zeros.unsqueeze(0).repeat(batch_size, 1, 1).to(x.device)  # (batch_size, ksize-1, input_dim=d_model)
#         x = torch.cat((zeros, x), dim=1)  # (batch_size, max_len+ksize-1, input_dim)
#         key = torch.index_select(x, 1, self.select_index[:self.ksize * l].to(x.device))  # (batch_size, ksize*l, input_dim)
#         key = key.reshape(batch_size, l, self.ksize, -1)  # (batch_size, l, ksize, input_dim)
#         return key
#
#
# class LocalRNNLayer(nn.Module):
#     "Encoder is made up of attconv and feed forward (defined below)"
#     def __init__(self, input_dim, output_dim, rnn_type, ksize, dropout):
#         super(LocalRNNLayer, self).__init__()
#         self.local_rnn = LocalRNN(input_dim, output_dim, rnn_type, ksize)
#         self.connection = SublayerConnection(output_dim, dropout)
#
#     def forward(self, x):
#         "Follow Figure 1 (left) for connections. Res connection."
#         x = self.connection(x, self.local_rnn)
#         return x
#
#
# class Block(nn.Module):
#     """a transformer encoder"""
#     def __init__(self, input_dim, output_dim, rnn_type, ksize, N, h, dropout):
#         super(Block, self).__init__()
#         # get model list
#         # rnn + res net + layer norm
#         self.layers = clones(
#             LocalRNNLayer(input_dim, output_dim, rnn_type, ksize, dropout), N)
#         self.connections = clones(SublayerConnection(output_dim, dropout), 2)
#         self.pooling = MHPooling(input_dim, h, dropout)  # Attention
#         self.feed_forward = PositionwiseFeedForward(input_dim, dropout)  # FFN全连接
#
#     def forward(self, x):
#         n, l, d = x.shape
#         # local rnn --> 利用短序列来提取序列特征，从而保证特征中具有方向以及距离等相关信息
#         for i, layer in enumerate(self.layers):
#             x = layer(x)
#         # multi-head attention
#         x = self.connections[0](x, self.pooling)
#         # ffn(non-liner)
#         x = self.connections[1](x, self.feed_forward)
#         return x
#
#
# class RTransformer(nn.Module):
#     def __init__(self, tag_size, dropout, d_model=256, rnn_type='GRU', ksize=10, n_level=1, n=1, h=4):
#         """
#         :param tag_size: 输出size
#         :param d_model: num_head*head_dim
#         :param rnn_type: 'RNN','LSTM','GRU'
#         :param ksize: kernel_size
#         :param n_level: num of encoders
#         :param n: num of local-rnn layers
#         :param h: num of heads
#         :param dropout: dropout prop
#         """
#         super(RTransformer, self).__init__()
#         N = n
#         self.d_model = d_model
#         layers = []
#         for i in range(n_level):
#             layers.append(
#                 Block(d_model, d_model, rnn_type, ksize, N=N, h=h, dropout=dropout))
#         self.forward_net = nn.Sequential(*layers)
#         self.hidden2tag = nn.Linear(d_model, tag_size)  # encoder的输出层
#
#     def forward(self, x, mask):
#         """
#         :param x: (batch_size, seq_len, d)
#         """
#         x = self.forward_net(x)
#         x = x * mask.unsqueeze(-1)
#         x = self.hidden2tag(x) * mask.unsqueeze(-1)  # (N, T, E)
#         return x
#
#
# class RTransformerEncoderModule(EncoderModule):
#     """
#     p = 0.3, lr = 1e-4  --> 0.827, 0.690, 1.140, 78.160, 0.962
#     p = 0.3, lr = 1e-3  --> 0.994, 0.711, 1.118, 77.071, 0.962
#     """
#     def __init__(self, params):
#         super(RTransformerEncoderModule, self).__init__(params=params)
#         self.proxy = RTransformer(
#             tag_size=self.params.encoder_output_size,  # encoder_output_size = config.hidden_size = 312
#             dropout=self.params.encoder_rtrans_dropout,  # 0.3
#             d_model=self.params.config.hidden_size,  # 312
#             rnn_type='GRU',
#             ksize=10,  # RNN执行过程中，子序列的长度大小
#             n_level=1,  # 整个结构Block重复多少次
#             n=1,  # 每个Block中的RNN重复多少次
#             h=4  # attention是几个头
#         )
#
#     def forward(self, input_feature, input_mask, **kwargs):
#         return self.proxy(input_feature, input_mask)
