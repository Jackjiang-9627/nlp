import torch
from torch import nn

from models.classify_models import SeqClassifyModule, OutputFCModule
from utils import Params


class SoftmaxSeqClassifyModule(SeqClassifyModule):
    def __init__(self, params: Params):
        super(SoftmaxSeqClassifyModule, self).__init__(params)
        # 定义特征提取模块
        if self.params.classify_fc_layers == 0:
            self.fc_layer = nn.Identity()
        else:
            layers = []
            input_unit = self.params.encoder_output_size
            for unit in self.params.classify_fc_hidden_size[:-1]:
                layers.append(OutputFCModule(params, input_unit, unit))
                input_unit = unit
            layers.append(nn.Linear(input_unit, self.params.num_labels))
            self.fc_layer = nn.Sequential(*layers)

        # 损失模块
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_feature, input_mask, labels=None, return_output=False, **kwargs):
        input_mask_weights = input_mask.unsqueeze(-1).to(input_feature.dtype)
        # 1. 获取每个Token对应的预测置信度
        feats = self.fc_layer(input_feature) * input_mask_weights  # [N,T,num_labels]
        # 2、损失或者预测的执行
        loss = None
        output = None
        if labels is not None:
            scores = torch.permute(feats, dims=[0, 2, 1])  # [N,T,num_labels] --> [N,num_labels,T]
            loss = self.loss_fn(scores, labels)
        if return_output:  # 返回预测结果
            pre_ids = torch.argmax(feats, dim=-1)  # [N,T,num_labels] -> [N,T]
            batch_size, max_len = input_mask.size()
            output = []
            for i in range(batch_size):
                real_len = input_mask[i].sum()
                output.append(list(pre_ids[i][:real_len].to('cpu').numpy()))
        return loss, output
