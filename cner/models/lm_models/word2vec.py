from torch import nn

from models.lm_models import LMModule
from utils import Params


class Word2VecLMModule(LMModule):
    def __init__(self, params: Params):
        super(Word2VecLMModule, self).__init__(params)
        #  可以自行实现gensim训练的word2vec向量作为embedding的初始_weight参数
        self.emb = nn.Embedding(
            num_embeddings=params.config.vocab_size, embedding_dim=params.config.hidden_size
        )

    def forward(self, input_ids, input_mask, **kwargs):
        """
        :param input_ids:  [N, T] --> [N, T, E]  float
        :param input_mask: [N, T] --> [N, T, 1] long
        :param kwargs: 额外参数
        :return: [N, T, E]
        """
        z = self.emb(input_ids)
        z = z * input_mask[..., None].to(z.dtype)
        return z
