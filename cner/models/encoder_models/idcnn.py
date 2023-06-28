import torch
from torch import nn

from models.encoder_models import EncoderModule
from utils import Params


class IDCNNEncoderModule(EncoderModule):
    """
    使用卷积来做NLP任务：主要利用N-Gram的思路，提取局部特征+膨胀dilation卷积提取大范围的特征（序列特征）
    不加dropout  --> 过拟合，val loss下降但acc不变f1很小
    p = 0.2, lr = 1e-4  --> 0.849, 0.729, 1.169, 67.573, 0.948
    p = 0.2, lr = 1e-3  --> f1很小
    p = 0.2, lr = 1e-5  --> f1很小
    p = 0.3, lr = 1e-3  --> 0.794, 0.706, 1.135, 77.464, 0.958   --> best!!!
    p = 0.3, lr = 1e-4  --> 0.908, 0.728, 1.142, 69.225, 0.951
    p = 0.4, lr = 1e-3  --> 1.055, 0.786, 1.164, 60.463, 0.942
    p = 0.4, lr = 1e-4  --> 0.771, 0.736, 1.157, 70.475, 0.952
    """
    def __init__(self, params: Params):
        super(IDCNNEncoderModule, self).__init__(params)
        # 单个block
        layers = []
        filters = self.params.encoder_idcnn_filters  # 卷积核数量，也就是卷积通道数 128
        # 遍历每层卷积层的参数
        for conv_params in self.params.encoder_idcnn_conv1d_params:
            kernel_size = conv_params.get('kernel_size', self.params.encoder_idcnn_kernel_size)
            dilation = conv_params.get('dilation', 1)
            layers.extend([
                nn.Conv1d(
                    in_channels=filters,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    padding='same',  # 步长为1时，输入输出大小一致（序列长度）
                    dilation=dilation
                ),
                nn.ReLU(),
                nn.BatchNorm1d(filters)
                # nn.LayerNorm()
            ])
        block = nn.Sequential(*layers)
        # 重复block
        blocks = []
        for i in range(self.params.encoder_idcnn_num_block):
            blocks.extend([
                nn.Dropout(p=self.params.encoder_dropout),  # 不加dropout过拟合，val loss下降但acc不变f1很小
                block,
                nn.ReLU(),
                nn.BatchNorm1d(filters)
            ])

        # network
        # 第一个全连接，将语言模型的输出特征(embedding_dims)合并
        self.fc_layer1 = nn.Linear(self.params.config.hidden_size, filters)
        # 卷积层
        self.conv_layer = nn.Sequential(*blocks)
        # 第二个全连接
        self.fc_layer2 = nn.Linear(filters, self.params.encoder_output_size, bias=True)

    def forward(self, input_feature, input_mask, **kwargs):
        """Note:将文本序列长度T当成L，将每个token的E维向量当成C个通道"""
        # 掩码变换维度
        input_mask_weights = input_mask.unsqueeze(-1).to(input_feature.dtype)  # [N, L] --> [N, L, 1]
        input_feature = self.fc_layer1(input_feature)

        # 卷积需要维度转换
        input_feature = torch.permute(input_feature, [0, 2, 1])  # [N, T, E] --> [N, E, T]([N,C,L])
        output_feature = self.conv_layer(input_feature)
        output_feature = output_feature.permute(0, 2, 1)  # [N,C,L] --> [N,L,C]
        encoder_feature = self.fc_layer2(output_feature)

        encoder_feature = encoder_feature * input_mask_weights
        return encoder_feature
