import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, num_class)

    def forward(self, inputs, lengths):
        embedding = self.embedding(inputs)
        # 使用pack_padded_sequence将变长序列打包
        x_pack = pack_padded_sequence(embedding, lengths=lengths, batch_first=True, enforce_sorted=False)
        hidden, (hn, cn) = self.lstm(x_pack)
        outputs = self.linear(hn[-1])
        log_probs = F.log_softmax(outputs, dim=-1)
        return log_probs
