import copy

from torch import nn


class FCModule(nn.Module):
    """一个全连接模块：线性层+激活+dropout，没有的话就用nn.Identity()替代"""
    def __init__(self, in_features, out_features, dropout=0.0, act=None):
        super(FCModule, self).__init__()
        # act为None的时候，采用默认激活函数, act为False的时候，设置为None，表示不需要激活函数
        if act is None:
            act = nn.ReLU()
        elif act is False:
            act = None
        else:
            act = act
        # act = nn.ReLU() if act is None else copy.deepcopy(act)
        # act = None if not act else act
        self.linear = nn.Linear(in_features, out_features)
        # 激活函数的选择
        self.act = nn.Identity() if act is None else act
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        return self.dropout(self.act(self.linear(x)))

class MLPModule(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=None,
                 dropout=0.0, act=None, decision_output=True):
        super(MLPModule, self).__init__()

        if hidden_features is None:  # 没有隐藏层
            hidden_features = []
        layers = []
        # 构造隐藏层序列
        for hidden_output_features in hidden_features:
            layers.append(FCModule(in_features, hidden_output_features, dropout=dropout, act=act))
            in_features = hidden_output_features  # 当前层的输出作为下一层的输入
        # 加入最后一个全连接模块做分类
        layers.append(
            FCModule(
                in_features, out_features,
                dropout=0.0 if decision_output else dropout,  # 决策层不需要dropout
                act=False if decision_output else act  # 决策层不需要激活函数
            )
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)