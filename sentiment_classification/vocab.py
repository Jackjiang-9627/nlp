from collections import defaultdict

class Vocab:
    """
    实现标记和索引之间的相互映射
    """
    def __init__(self, tokens=None):
        self.idx_to_token = []  #
        self.token_to_idx = {}

        if tokens is not None:
            if '<unk>' not in tokens:
                tokens = tokens + ['<unk>']
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
            self.unk = self.token_to_idx['<unk>']

    @classmethod
    def build(cls, text, min_freq=1, reserved_tokens=None):
        token_freqs = defaultdict(int)
        for sentence in text:
            for token in sentence:
                token_freqs[token] += 1
        uniq_tokens = ['<unk>'] + (reserved_tokens if reserved_tokens else [])
        uniq_tokens += [token for token, freq in token_freqs.items() if freq >= min_freq and token != '<unk>']
        return cls(uniq_tokens)

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token):  # 查找索引值
        # 不存在则返回<unk>对应的索引值(0)
        return self.token_to_idx.get(token, self.unk)

    def convert_tokens_to_ids(self, tokens):
        return [self[token] for token in tokens]

    def convert_idx_to_tokens(self, indices):
        return [self.token_to_idx[index] for index in indices]

