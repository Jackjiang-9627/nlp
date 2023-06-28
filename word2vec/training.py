"""
模型训练相关代码
"""
import os.path
import pickle

import torch
from torch import optim
from torch.utils.data import DataLoader

from word2vec.data_process import Vocab, creat_dir, Word2VecDataset
from word2vec.model import SkipGramModule, CBOWModule

def restore_network(net, model_dir):
    if not os.path.exists(model_dir):
        return 0  # model_dir目录下没有文件，训练还未开始
    files = os.listdir(model_dir)  # 获取model_dir目录下的文件，以列表返回文件名
    files.sort()  # 升序排列文件名
    if len(files) == 0:
        return 0
    # 获取最后一个模型文件的完整路径
    file = os.path.join(model_dir, files[-1])
    # 加载恢复模型
    ckpt = torch.load(file, map_location='cpu')
    # 加载恢复model
    net.load_state_dict(ckpt['model'].state_dict())
    return ckpt['epoch'] + 1


def run(vocab_file, output_dir, train_file, batch_size=1000,
        cbow=True, embedding_dim=128, num_negative_sample=20,
        total_epoch=10, save_internal_epoch=1):
    """
    vocab_file:词表路径
    output_dir：模型保存目录
    train_file：训练数据文件路径，用于构造数据集
    cbow：True表示采用CBOW模型，False表示采用SkipGram模型
    total_epoch：一共训练10个epoch
    save_internal_epoch：间隔1个epoch保存一次
    """
    # 0. 属性准备
    vocab = Vocab.load(vocab_file)
    output_model_dir = os.path.join(output_dir, 'model')  # 模型保存的目录
    creat_dir(output_model_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1.数据构建
    dataset = Word2VecDataset(train_file)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 给定底层使用几个线程进行数据加载，0表示使用训练程序这个主线程进行加载
        collate_fn=None  # 给定如何将多个样本组合成一个批次对象，默认一般情况下不需要给定
    )
    # 2. 模型对象、损失函数、优化器
    model_class = CBOWModule if cbow else SkipGramModule
    net = model_class(
        num_embeddings=len(vocab),
        embedding_dim=embedding_dim,
        k=num_negative_sample,
        all_label_weights=vocab.fetch_negative_sampling_token_frequency()
    ).to(device)

    train_op = optim.SGD(net.parameters(), lr=0.1, weight_decay=0.01)
    # 如果程序意外停止，模型从保存文件中恢复，程序会自动计算新的start_epoch
    start_epoch = restore_network(net, model_dir=output_model_dir)

    # 3. 迭代训练
    batch_idx = 0
    # total_epoch = total_epoch + start_epoch  # todo:?

    for epoch in range(start_epoch, total_epoch):
        # 训练
        net.train()
        for context_words, central_words in dataloader:
            # 将数据转移到device上，数据要在同一类型的设备上， 不能模型参数在gpu，输入数据在cpu，会报错
            context_words = context_words.to(device)
            central_words = central_words.to(device)
            # 前向过程，计算损失
            loss = net(context_words, central_words)
            # 反向
            train_op.zero_grad()
            loss.backward()
            train_op.step()
            # 日志打印
            if batch_idx % 100 == 0:
                print(f'Epoch:{epoch}/{total_epoch} Batch:{batch_idx} Train Loss:{loss.item():.3f}')
            batch_idx += 1
        # 评估
        eval_batch = 0
        net.eval()
        for context_words, central_words in dataloader:
            context_words = context_words.to(device)
            central_words = central_words.to(device)
            loss = net(context_words, central_words, negative_sampling=False)
            # 日志打印
            print(f'Epoch:{epoch}/{total_epoch} Batch:{batch_idx} Eval Loss:{loss.item():.3f}')
            eval_batch += 1
            if eval_batch > 20:
                break

        # 模型保存
        if (epoch % save_internal_epoch == 0) or epoch == total_epoch - 1:
            torch.save(
                obj={'model': net, 'epoch': epoch},
                f=os.path.join(output_model_dir, f'model_{epoch:04d}.pkl')
            )
    # 4. 保存词表
    vocab.save(os.path.join(output_dir, 'vocab.pkl'))
    # 保存embedding层的权重矩阵即embedding_table
    with open(os.path.join(output_dir, 'embedding_table.pkl'), 'wb') as writer:
        pickle.dump(net.weight.to('cpu').detach().numpy(), writer)


if __name__ == '__main__':
    run(
        vocab_file='./output/datas/vocab.pkl',
        output_dir='./output',
        train_file='./output/datas/红楼梦_training.txt',
        total_epoch=10
    )
