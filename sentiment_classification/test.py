import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data_processor import load_sentence_polarity, BowDataset, collate_fn_mlp, collate_fn_cnn, \
    collate_fn_lstm
from .model_cnn import CNN
from .model_mlp import MLP
from .model_lstm import LSTM


def test_mlp(collate_fn):
    # 超参数设置
    embedding_dim = 128
    hidden_dim = 256
    num_class = 2
    batch_size = 32
    num_epoch = 100
    # 加载数据
    train_data, test_data, vocab = load_sentence_polarity()
    train_dataset = BowDataset(train_data)
    test_dataset = BowDataset(test_data)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=False
    )

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(len(vocab), embedding_dim, hidden_dim, num_class)
    model.to(device)

    # 训练过程
    nll_loss = nn.NLLLoss()  # negative  log likelihood loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(num_epoch):
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f'Training Epoch {epoch}'):
            inputs, offsets, targets = [x.to(device) for x in batch]
            log_probs = model(inputs, offsets)
            loss = nll_loss(log_probs, targets)
            optimizer.step()
            total_loss += loss.item()
    print(f"Loss: {total_loss:.2f}")

    # 测试过程
    acc = 0
    for batch in tqdm(test_dataloader, desc=f'Testing'):
        inputs, offsets, targets = [x.to(device) for x in batch]
        with torch.no_grad():
            output = model(inputs, offsets)
            acc += (output.argmax(dim=1) == targets).sum().item()

    # 输出在测试集上的准确率
    print(f'ACC: {acc / len(test_dataloader):.2f}')


def test_cnn(collate_fn):
    # 超参数设置
    embedding_dim = 128
    kernel_size = 3
    out_channels = 100
    num_class = 2
    batch_size = 32
    num_epoch = 100
    # 加载数据
    train_data, test_data, vocab = load_sentence_polarity()
    train_dataset = BowDataset(train_data)
    test_dataset = BowDataset(test_data)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=False
    )

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN(len(vocab), embedding_dim, kernel_size=3, out_channels=100, num_class=2)
    model.to(device)

    # 训练过程
    nll_loss = nn.NLLLoss()  # negative  log likelihood loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(num_epoch):
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f'Training Epoch {epoch}'):
            inputs, targets = [x.to(device) for x in batch]
            log_probs = model(inputs)
            loss = nll_loss(log_probs, targets)
            optimizer.step()
            total_loss += loss.item()
    print(f"Loss: {total_loss:.2f}")

    # 测试过程
    acc = 0
    for batch in tqdm(test_dataloader, desc=f'Testing'):
        inputs, *_, targets = [x.to(device) for x in batch]
        with torch.no_grad():
            output = model(inputs)
            acc += (output.argmax(dim=1) == targets).sum().item()

    # 输出在测试集上的准确率
    print(f'ACC: {acc / len(test_dataloader):.2f}')


def test_lstm(collate_fn):
    # 超参数设置
    embedding_dim = 128
    hidden_dim = 256
    num_class = 2
    batch_size = 32
    num_epoch = 100
    # 加载数据
    train_data, test_data, vocab = load_sentence_polarity()
    train_dataset = BowDataset(train_data)
    test_dataset = BowDataset(test_data)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=False
    )

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTM(len(vocab), embedding_dim, hidden_dim, num_class)
    model.to(device)

    # 训练过程
    nll_loss = nn.NLLLoss()  # negative  log likelihood loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(num_epoch):
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f'Training Epoch {epoch}'):
            inputs, lengths, targets = [x.to(device) for x in batch]
            log_probs = model(inputs, lengths.to('cpu'))  # lengths.to('cpu'):报错
            loss = nll_loss(log_probs, targets)
            optimizer.step()
            total_loss += loss.item()
    print(f"Loss: {total_loss:.2f}")

    # 测试过程
    acc = 0
    for batch in tqdm(test_dataloader, desc=f'Testing'):
        inputs, lengths, targets = [x.to(device) for x in batch]
        with torch.no_grad():
            output = model(inputs, lengths.to('cpu'))
            acc += (output.argmax(dim=1) == targets).sum().item()

    # 输出在测试集上的准确率
    print(f'ACC: {acc / len(test_dataloader):.2f}')


if __name__ == '__main__':
    # test_mlp(collate_fn=collate_fn_mlp)
    # test_rnn(collate_fn_cnn)
    test_lstm(collate_fn_lstm)
