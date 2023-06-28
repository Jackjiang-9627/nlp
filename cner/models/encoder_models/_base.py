from torch import nn

from utils import Params


class EncoderModule(nn.Module):
    def __init__(self, params: Params):
        super(EncoderModule, self).__init__()
        self.params = params

    def forward(self, input_feature, input_mask, **kwargs):
        """
        从词向量中提取特征
            Note:
                E1 == self.params.config.hidden_size
                E2 == self.params.config.encoder_output_size
        :param input_feature: [N, T, E1] float类型
        :param input_mask: [N, T]  long类型
        :param kwarge: 额外参数，子类中可能需要用到
        :return: 最终输出每个token对应的新的特征向量 [N, T, E2] float类型
        """
        raise NotImplementedError("该方法在当前子类中未实现!")
