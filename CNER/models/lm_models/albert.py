import torch
from torch import nn
from transformers import AlbertModel, BertModel

from models.lm_models import LMModule
from utils import Params


# noinspection PyBroadException
class ALBertLMModule(LMModule):
    def __init__(self, params: Params):
        super(ALBertLMModule, self).__init__(params)
        try:  # bert模型迁移
            self.bert = AlbertModel.from_pretrained(
                config=self.param.config,
                pretrained_model_name_or_path=self.params.bert_root_dir  # 参数恢复路径
            )
            print(1)
        except Exception as _:  # 迁移失败，初始化
            self.bert = AlbertModel(config=self.params.config, add_pooling_layer=False)
            self.freeze_params = False
        # 融合层数，总共有num_hidden_layers层，选后lm_fusion_layers=4层
        self.fusion_layers = int(min(self.params.config.num_hidden_layers, self.params.lm_fusion_layers))
        # 动态权重系数，决定每层输出的权重
        self.dym_weight = nn.Parameter(torch.ones(self.fusion_layers, 1, 1, 1))
        nn.init.xavier_normal_(self.dym_weight)  # 参数初始化

    def freeze_model(self):
        if not self.freeze_params:  # 不冻结直接返回空，什么也不做
            return
        print("冻结AlBert语言模型参数!")
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, input_mask, **kwargs):
        outputs = self.bert(
            input_ids,
            attention_mask=input_mask,  # todo：input_mask在此处的效果
            output_hidden_states=True  # 是否返回每一层的结果值
        )  # [N,T] --> [N,T,E]
        # 需用将albert每一层的返回进行加权合并
        # 将list/tuple形式的tensor合并成一个tensor对象, [fusion_layers, N,T,E]
        hidden_stack = torch.stack(outputs.hidden_states[-self.fusion_layers:], dim=0)  # 从后往前
        hidden_stack = hidden_stack * self.dym_weight
        z = torch.sum(hidden_stack, dim=0)  # [fusion_layers, N,T,E] --> [N,T,E]
        z = z * input_mask[..., None].to(z.dtype)
        return z
