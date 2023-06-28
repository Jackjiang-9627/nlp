from torch import nn
from utils import Params

# noinspection PyUnresolvedReferences
from models.lm_models import *
# noinspection PyUnresolvedReferences
from models.encoder_models import *
# noinspection PyUnresolvedReferences
from models.classify_models import *

class NERTokenClassification(nn.Module):
    def __init__(self, params: Params):
        super(NERTokenClassification, self).__init__()
        self.with_encoder = params.with_encoder
        # language model
        self.lm_layer = eval(params.lm_layer_name)(params)
        # encoder model
        if self.with_encoder:
            self.encoder_layer = eval(params.encoder_layer_name)(params)
        else:
            self.encoder_layer = nn.Identity()
        # classify model
        self.classify_layer = eval(params.classify_layer_name)(params)
        # 如果需要，冻结模型
        if params.lm_freeze_params:
            self.lm_layer.freeze_model()

    def forward(self, input_ids, input_masks, labels, return_output=False):
        z = self.lm_layer(input_ids, input_masks)
        if self.with_encoder:
            z = self.encoder_layer(z, input_masks)
        else:
            z = self.encoder_layer(z)
        z = self.classify_layer(z, input_masks, labels=labels, return_output=return_output)
        return z

def build_model(params: Params):
    model = NERTokenClassification(params)
    return model
