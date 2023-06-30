import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
        super(MLP, self).__init__()
        # EmbeddingBag: 首先将不定长序列拼接起来，用偏移向量（offset）记录每个序列的起始位置
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.activate = F.relu
        self.linear2 = nn.Linear(hidden_dim, num_class)

    def forward(self, inputs, offsets):
        embedding = self.embedding(inputs, offsets)
        hidden = self.activate(self.linear1(embedding))
        outputs = self.linear2(hidden)
        log_probs = F.log_softmax(outputs, dim=1)
        return log_probs


# if __name__ == '__main__':
#     mlp = MLP(vocab_size=8, embedding_dim=3, hidden_dim=5, num_class=2)
#     inputs = torch.tensor([[0, 1, 2, 1], [4, 6, 6, 7]], dtype=torch.long)
#     outputs = mlp(inputs)
#     print(outputs)
