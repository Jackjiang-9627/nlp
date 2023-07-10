import torch.nn as nn
class TokenEmbeddingModule(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(TokenEmbeddingModule, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.output_size = embedding_dim

    def forward(self, x):
        return self.embedding(x)  # [N, T] --> [N, T, E]
