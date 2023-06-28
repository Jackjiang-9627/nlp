import torch
from torch import nn

from models.encoder_models import EncoderModule
from utils import Params


class BiLSTMEncoderModule(EncoderModule):
    def __init__(self, params: Params):
        super(BiLSTMEncoderModule, self).__init__(params)
        self.dropout = nn.Dropout(p=params.encoder_dropout)
        self.lstm_layer = nn.LSTM(
            input_size=self.params.config.hidden_size,
            hidden_size=self.params.encoder_lstm_hidden_size,
            num_layers=self.params.encoder_lstm_layers,
            dropout=0 if self.params.encoder_lstm_layers == 1 else self.dropout,
            bidirectional=True,
            batch_first=True
        )
        lstm_output_size = self.params.encoder_lstm_hidden_size * 2  # 双向lstm，在最后一维拼接
        self.layer_norm = nn.LayerNorm(lstm_output_size) \
            if self.params.encoder_lstm_with_ln else nn.Identity()
        self.fc_layer = nn.Linear(lstm_output_size, self.params.encoder_output_size, bias=True)

    def forward(self, input_feature, input_mask, **kwargs):
        """
        注释掉的是batch_first=False时的代码
        :param input_feature: [N, T, E1]  --> [T, N, E1]
        :param input_mask:    [N, T] long --> [T, N] long --> [T, N, 1] float
        :param kwarge:
        :return:
        """
        # input_feature = torch.permute(input_feature, dims=[1, 0, 2])
        # input_mask = torch.permute(input_mask, dims=[1, 0])
        input_mask_weights = input_mask.unsqueeze(-1).to(input_feature.dtype)
        _, max_len = input_mask.size()  # max_len, _ = input_mask.size()
        input_feature = self.dropout(input_feature)

        # 1、LSTM提取序列特征信息
        """
        pad_sequence:填充序列 + pack_padded_sequence = pack_sequence  <--> pad_packed_sequence
        """
        embed = nn.utils.rnn.pack_padded_sequence(
            input_feature,
            input_mask.sum(1).long().to('cpu'),  # input_mask.sum(0).long().to('cpu')
            enforce_sorted=False,
            batch_first=True  # batch_first=False
        )
        lstm_output, _ = self.lstm_layer(embed)  # [T, N, E1]
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output, total_length=max_len, batch_first=True)  # batch_first=False
        lstm_output = lstm_output * input_mask_weights

        # 2、Norm层防止模型过拟合、加快训练速度
        lstm_output = self.layer_norm(lstm_output)

        # 3、全连接特征融合
        encoder_feature = self.fc_layer(lstm_output)
        if self.fc_layer.bias is not None:  # bias将会使pad处的位置不为0
            encoder_feature *= input_mask_weights
        # encoder_feature = torch.permute(encoder_feature, dims=[1, 0, 2])  # [N, T, E2]
        return encoder_feature
