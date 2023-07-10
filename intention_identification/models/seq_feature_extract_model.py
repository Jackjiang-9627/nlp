import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence


class RNNSeqFeatureExtractModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1,
                 batch_first=True, dropout=0.0, bidirectional=False,
                 output_type="all_mean"
                 ):
        super(RNNSeqFeatureExtractModule, self).__init__()
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.rnn_layer = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=self.batch_first,
            dropout=dropout,
            bidirectional=self.bidirectional
        )
        # 隐层状态输出形状，是否双向
        self.hidden_size = hidden_size * (2 if bidirectional else 1)
        # 对rnn输出层做求和或平均
        self.output_is_sum_mean = output_type in ['all_sum', 'all_mean']
        self.output_is_mean = output_type == 'all_mean'

    def forward(self, input_features, seq_lengths):
        # 输入的dataset类型为：已做填充截断的的tensor对象，序列实际长度信息 --> 构建一个PackedSequence对象
        # 打包后的序列
        seq_packed = pack_padded_sequence(
            input_features, seq_lengths.to('cpu'),
            batch_first=self.batch_first,
            enforce_sorted=False  # 不按序列长度进行排序
        )
        # 此时，rnn_layer的输入变成PackedSequence对象，而不是tensor对象,两种对象都可以作为nn.RNN的输入
        # rnn_output为最后一层所有时刻的隐状态  [N, T, E]
        # hidden_out为最后一个时刻的所有层的隐状态 [D * num_layers, N, E]
        """lstm的输出相比rnn、gru多了一个状态c，所以另创建一个函数来前向传播，更改lstm只需重构这个函数"""
        # rnn_output, hidden_out = self.rnn_layer(seq_packed)
        rnn_output, hidden_out = self.rnn_layer(seq_packed)
        # 分类需要将[N, T, E] --> [N, E]
        # 用rnn的输出特征做分类
        if self.output_is_sum_mean:
            # 反过来解析PackedSequence对象 --> 填充的tensor对象 + 样本的实际长度信息
            seq_unpacked, lens_unpacked = pad_packed_sequence(
                rnn_output,
                self.batch_first,
                padding_value=0.0  # 填充值，默认0.0
            )
            output = torch.sum(seq_unpacked, dim=1).to('cpu')  # [N, T, E] --> [N, E]
            if self.output_is_mean:
                output = output / lens_unpacked[:, None]  # [N,E] / [N,1] ->  [N,E]
        else:  # 用rnn的隐状态做分类
            if self.bidirectional:
                # 双向rnn，需要将隐状态拼接  [-1, -2]为最后一层的双向隐状态  [N,E],[N,E] --> [N,2E]
                output = torch.concat([hidden_out[-1, ...], hidden_out[-2, ...]], dim=1)
            else:  # 取最后一层的隐状态来分类
                output = hidden_out[-1, ...]  # [num_layer,N,E] -> [N,E]
        return output.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # 返回的批次中的每个E维向量用于该句子分类

    def fetch_rnn_features(self, seq_packed: PackedSequence):
        return self.rnn_layer(seq_packed)

class GRUSeqFeatureExtractModule(RNNSeqFeatureExtractModule):
    def __init__(self, input_size, hidden_size, num_layers=1,
                 batch_first=True, dropout=0.0, bidirectional=False,
                 output_type="all_mean"
                 ):
        super(GRUSeqFeatureExtractModule, self).__init__(
            input_size, hidden_size, num_layers, batch_first, dropout, bidirectional, output_type
        )
        # 相当于做了一个覆盖,将nn.RNN换成nn.GRU, forward函数与RNN完全一致
        self.rnn_layer = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=self.batch_first,
            dropout=dropout,
            bidirectional=self.bidirectional
        )

class LSTMSeqFeatureExtractModule(RNNSeqFeatureExtractModule):
    def __init__(self, input_size, hidden_size, output_type="all_mean",
                 num_layers=1, batch_first=True, dropout=0.0, bidirectional=False,
                 proj_size=0  # 0：不做投影; >0:将隐状态投影到相应大小
                 ):
        super(LSTMSeqFeatureExtractModule, self).__init__(
            input_size, hidden_size, num_layers, batch_first, dropout, bidirectional, output_type
        )

        # 相当于一个覆盖,将nn.RNN换成nn.LSTM
        self.rnn_layer = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=batch_first,
            dropout=dropout, bidirectional=bidirectional,
            proj_size=proj_size
        )

    def fetch_rnn_features(self, seq_packed: PackedSequence):
        """重构rnn_layer的输出，保持一致性，均为rnn_output, hidden_out"""
        rnn_output, (hidden_out, _) = self.rnn_layer(seq_packed)
        return rnn_output, hidden_out
