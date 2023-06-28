from torch import nn

from utils import Params


class LMModule(nn.Module):
    def __init__(self, params: Params):
        super(LMModule, self).__init__()
        self.params = params
        self.freeze_params = params.lm_freeze_params

    def forward(self, input_ids, input_mask, **kwargs):
        """
        获取输入对应的每个token的特征向量
        :param input_ids:  token对应的id列表 [N, T] long类型
        :param input_mask: 掩码  [N, T] long类型
        :param kwargs: 额外参数，子类中需要用到
        :return: [N, T, E] E就是 self.params.config.hidden_size float类型
        """
        raise NotImplementedError("该方法在当前子类中未实现")

    def freeze_model(self):
        """
        冻结模型，子类具体实现，默认不进行任何冻结操作
        :return: 提示当前子类模型未冻结模型参数
        """
        print(f'当前实现不冻结模型参数{__class__}')

