import torch.nn as nn
from torch.nn import functional as F


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, kernel_size, out_channels, num_class):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # [N, T] --> [N, T, E]
        # in_channels:对应词向量维度， out_channels:对应卷积核个数， kernel_size：卷积核宽度
        # 输入形状为(batch, in_channels, seq_len) -->  输出形状为(batch, out_channels, seq_len)
        self.cnn = nn.Conv1d(embedding_dim, out_channels, kernel_size, padding=1)  # [N, E, T] --> [N, O, T]
        self.activate = F.relu
        self.linear = nn.Linear(out_channels, num_class)  # [N, O, T] --> [N, C, T]

    def forward(self, inputs):
        embedding = self.embedding(inputs)
        convolution = self.activate(self.cnn(embedding.permute(0, 2, 1)))  # [N, T, E] --> [N, E, T]
        pooling = F.max_pool1d(convolution, kernel_size=convolution.shape[2])  # [N, E, T] --> [N,T, 1]
        outputs = self.linear(pooling.squeeze(dim=2))
        log_probs = F.log_softmax(outputs, dim=1)
        return log_probs

