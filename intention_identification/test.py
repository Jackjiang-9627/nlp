import torch

token_vocab_file = "intention_identification/datas/output/vocab.pkl"
torch.load(token_vocab_file, map_location='cpu')