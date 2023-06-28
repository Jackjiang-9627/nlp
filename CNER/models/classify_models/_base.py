from torch import nn

from utils import Params


class SeqClassifyModule(nn.Module):
    def __init__(self, params: Params):
        super(SeqClassifyModule, self).__init__()
        self.params = params

    def forward(self, input_feature, input_mask, labels=None, return_output=False, **kwargs):
        """
        最终的决策输出/分类输出，最终返回的是各个类别的置信度、损失值等数据
        NOTE:
                E2 == self.param.encoder_output_size
        :param input_feature: [N,T,E2] N个样本，每个样本T个时刻/token，每个时刻/token对应一个E2维的向量
        :param input_mask: 每个token是否是实际token(是不是填充值),1表示实际值，0表示填充值， [N,T], long类型
        :param labels: 实际标签 [N,T]
        :param return_output: 是否返回预测输出值，默认为False，表示返回的output为None
        :param kwargs: 额外参数，可能子类中需要使用到
        :return: loss, output
            loss:就是一个损失的tensor标量，当labels未给定的时候，loss为None
            output:List[List[int]] 每个样本、每个token对应的预测类别id
        """
        raise NotImplementedError("该方法在当前子类中未实现.")

class OutputFCModule(nn.Module):
    def __init__(self, param, input_features, output_features):
        super(OutputFCModule, self).__init__()
        self.dropout = nn.Dropout(param.classify_fc_dropout)
        self.linear = nn.Linear(input_features, output_features)
        self.layer_norm = nn.LayerNorm(output_features)

    def forward(self, x):
        return self.layer_norm(self.linear(self.dropout(x)))