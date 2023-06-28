"""1定义模型相关code"""
import torch
import torch.nn as nn


class RNNTextClassifyModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1, batch_first=True, dropout=0.0,
                 bidirectional=True):
        super(RNNTextClassifyModel, self).__init__()
        self.emb_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.rnn_layer = nn.RNN(
            input_size=10,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=False
        )
        if bidirectional:
            self.reverse_rnn_layer = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=False
            )


    def forward(self, seqs, reverse_seqs, seq_lengths):
        """
        分类模型的前向过程
        :param seqs: [N, T]:N表示当前批次有N个样本，T表示每个样本有T个token
        :param seq_lengths: [N] 表示每个样本的实际长度
        :return:
        """
        x1 = self.emb_layer(seqs)  # 每个token获取该token对应的特征向量值
        x2 = self.emb_layer(reverse_seqs)
        x1, _ = self.rnn_layer(x1)  # rnn的正向过程
        x2, _ = self.rnn_layer(x2)  # rnn的反向过程
        print('x')
        pass


if __name__ == '__main__':
    token_vocab = torch.load('./datas/output/vocab.pkl')
    model = RNNTextClassifyModel(
        vocab_size=len(token_vocab),
        embedding_dim=128,
        hidden_size=128,
        num_layers=2,
        bidirectional=True
    )
    seq_idxes = torch.tensor([[285, 2124, 123, 6, 29, 1, 1828, 4586, 13, 205, 0, 0,
                               0, 0, 0],
                              [619, 598, 333, 396, 1, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0],
                              [105, 21, 36, 228, 158, 1530, 1531, 1, 287, 6, 288, 0,
                               0, 0, 0],
                              [131, 132, 921, 1, 2230, 1002, 0, 0, 0, 0, 0, 0,
                               0, 0, 0],
                              [190, 532, 279, 1091, 1, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0],
                              [190, 154, 1, 112, 1318, 865, 448, 2199, 0, 0, 0, 0,
                               0, 0, 0],
                              [57, 1, 21, 1, 2096, 1, 21, 1702, 1, 1, 21, 1702,
                               1, 57, 1],
                              [105, 21, 83, 190, 1062, 3258, 292, 420, 0, 0, 0, 0,
                               0, 0, 0]], dtype=torch.int32)
    reverse_seq_idxes = torch.tensor([
            [205, 13, 4586, 1828, 1, 29, 6, 123, 2124, 285, 0, 0,
             0, 0, 0],
            [1, 396, 333, 598, 619, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0],
            [288, 6, 287, 1, 1531, 1530, 158, 228, 36, 21, 105, 0,
             0, 0, 0],
            [1002, 2230, 1, 921, 132, 131, 0, 0, 0, 0, 0, 0,
             0, 0, 0],
            [1, 1091, 279, 532, 190, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0],
            [2199, 448, 865, 1318, 112, 1, 154, 190, 0, 0, 0, 0,
             0, 0, 0],
            [1, 57, 1, 1702, 21, 1, 1, 1702, 21, 1, 2096, 1,
             21, 1, 57],
            [420, 292, 3258, 1062, 190, 83, 21, 105, 0, 0, 0, 0,
             0, 0, 0]], dtype=torch.int32)
    seq_lengths = torch.tensor([8, 8, 10, 8, 8, 11, 10, 5], dtype=torch.int32)
    r = model(seq_idxes, reverse_seq_idxes, seq_lengths)
    print(r)
