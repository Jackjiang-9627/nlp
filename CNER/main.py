import logging
import os
from itertools import chain

import torch

import utils
from dataloader import NERDataLoader
from models.model import build_model
from train import train_and_evaluate


def run(ex_index, bert_root_dir, input_params):
    # 设定GPU运行的device id列表
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

    """1、开始 构建参数"""
    logging.info("Start build params...")
    # ex_index:实验生成文件的id --> run_(ex_index)
    # LSTM + SOFTMAX -->
    # 1：bert的词向量  2：bert的词向量（GPU） 3：albert预训练 4：bert预训练 5：roberta预训练  6：nezha(train_batch=16)
    # CNN + SOFTMAX -->
    # 7:roberta   8：rtransformer
    # 更改预训练文件的目录和语言模型的名称
    params = utils.build_params(ex_index, bert_root_dir, input_params)
    utils.set_logger(save=True, log_path=params.params_path / "train.log")
    # 打印参数信息
    logging.info(f'Params:\n{params}')

    """2、 加载数据"""
    dataloader = NERDataLoader(params)
    train_loader = dataloader.get_dataloader('train')
    val_loader = dataloader.get_dataloader('val')

    """3、构建模型"""
    logging.info('Start build train model...')
    model = build_model(params)
    model.to(device=params.device)  # 模型参数得跟输入的数据在同一个设备上，否则报错
    logging.info(f"Model:\n{model}")
    if params.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    """4、构建优化器"""
    logging.info("Start build optimizer...")
    # 训练中的总参数更新次数 = 总批次 // 参数更新间隔批次 * 总循环次数
    total_train_epoch = len(train_loader) // params.gradient_accumulation_steps * params.epoch_num
    optimizer = utils.build_optimizer(model, params, total_train_epoch)

    """5、训练&评估"""
    logging.info(f"Start train for {params.epoch_num} epochs...")
    train_and_evaluate(model, optimizer, train_loader, val_loader, params)
    logging.info(f"{params.lm_layer_name}-lr:{params.lm_tuning_lr}\n"
                 f"{params.encoder_layer_name}-lr:{params.encoder_lr}-dropout:{params.encoder_dropout}\n"
                 f"{params.classify_layer_name}-lr:{params.classify_lr}")
    logging.info("Completed training!")

def auto_run_params():
    """自动化训练参数配置"""
    ex_ids = list(range(1, 31))
    bert_root_dirs = [r'./pre_train_models/bert', r'./pre_train_models/bert', r'./pre_train_models/albert',
                      r'./pre_train_models/roberta', r'./pre_train_models/nezha']
    lm_names = ['Word2VecLMModule', 'BertLMModule', 'ALBertLMModule', 'RoBertaLMModule', 'NEZHALMModule']
    encoder_names = ['BiLSTMEncoderModule', 'IDCNNEncoderModule', 'RTransformerEncoderModule']
    classify_names = ['SoftmaxSeqClassifyModule', 'CRFSeqClassifyModule']
    params = []
    for bert_root_dir, lm_name in zip(bert_root_dirs, lm_names):
        for encoder_name in encoder_names:
            for classify_name in classify_names:
                param = {
                    'data_dir': r'.\datas\sentence_tag',
                    'bert_root_dir': bert_root_dir,
                    'lm_layer_name': lm_name,  # 不定义则采用'Word2VecLMModule'
                    'encoder_layer_name': encoder_name,  # 不定义则采用'BiLSTMEncoderModule'
                    'classify_layer_name': classify_name,  # 不定义则采用'SoftmaxSeqClassifyModule'
                    # 'encoder_lr': encoder_lr,
                    # 'train_batch_size': train_batch_size,
                    # 'lm_freeze_params': True,  # 不定义则不冻结语言模型参数
                    # 'with_encoder': False,  # 不定义则默认有encoder层
                    # 'encoder_output_size': 27,  # 不定义则采用config.hidden_size
                    # 'classify_lr': 1e-3,
                }
                if encoder_name == 'IDCNNEncoderModule':
                    param['encoder_lr'] = 1e-3
                if lm_name == 'NEZHALMModule' or encoder_name == 'RTransformerEncoderModule':
                    param['train_batch_size'] = 16
                params.append(param)

    bert_root_dirs = list(chain(*map(lambda i: [i] * 6, bert_root_dirs)))
    for ex_index, bert_root_dir, param in zip(ex_ids, bert_root_dirs, params):
        print((ex_index, bert_root_dir, param))
        run(ex_index, bert_root_dir, param)

def run_NEZHA():
    """nezha系列"""
    ex_ids = list(range(25, 31))
    bert_root_dirs = [r'./pre_train_models/nezha'] * 6
    bert_root_dir = bert_root_dirs[0]
    params = []
    encoder_names = ['BiLSTMEncoderModule', 'IDCNNEncoderModule', 'RTransformerEncoderModule']
    classify_names = ['SoftmaxSeqClassifyModule', 'CRFSeqClassifyModule']
    lm_name = 'NEZHALMModule'
    for encoder_name in encoder_names:
        encoder_lr = 1e-3 if encoder_name == 'IDCNNEncoderModule' else 1e-4
        for classify_name in classify_names:
            params.append({
                'data_dir': r'.\datas\sentence_tag',
                'bert_root_dir': bert_root_dir,
                'lm_layer_name': lm_name,  # 不定义则采用'Word2VecLMModule'
                'encoder_layer_name': encoder_name,  # 不定义则采用'BiLSTMEncoderModule'
                'classify_layer_name': classify_name,  # 不定义则采用'SoftmaxSeqClassifyModule'
                'encoder_lr': encoder_lr,
                'train_batch_size': 16,
                # 'lm_freeze_params': True,  # 不定义则不冻结语言模型参数
                # 'with_encoder': False,  # 不定义则默认有encoder层
                # 'encoder_output_size': 27,  # 不定义则采用config.hidden_size
                # 'classify_lr': 1e-3,
            })

    for ex_index, bert_root_dir, param in zip(ex_ids, bert_root_dirs, params):
        # print((ex_index, bert_root_dir, param))
        run(ex_index, bert_root_dir, param)

def run_other():
    """
    run30之外的情况：
    run5_1: dropout=0
    """
    """自动化训练参数配置"""
    ex_index = '31'
    bert_root_dir = r'./pre_train_models/roberta'
    # lm_names = ['Word2VecLMModule', 'BertLMModule', 'ALBertLMModule', 'RoBertaLMModule', 'NEZHALMModule']
    # encoder_names = ['BiLSTMEncoderModule', 'IDCNNEncoderModule', 'RTransformerEncoderModule']
    # classify_names = ['SoftmaxSeqClassifyModule', 'CRFSeqClassifyModule']
    param = {
        'data_dir': r'.\datas\sentence_tag',
        'bert_root_dir': bert_root_dir,
        'lm_layer_name': 'RoBertaLMModule',  # 必须设定
        'encoder_layer_name': 'BiLSTMEncoderModule',  # 不定义则采用'BiLSTMEncoderModule'
        'classify_layer_name': 'CRFSeqClassifyModule',  # 不定义则采用'SoftmaxSeqClassifyModule'
        'encoder_dropout': 0.3,  # 默认0.3
        'encoder_lr': 1e-5,
        # 'train_batch_size': train_batch_size,
        # 'lm_freeze_params': True,  # 不定义则不冻结语言模型参数
        # 'with_encoder': False,  # 不定义则默认有encoder层
        # 'encoder_output_size': 27,  # 不定义则采用config.hidden_size
        # 'classify_lr': 1e-3,
    }
    run(ex_index, bert_root_dir, param)


if __name__ == '__main__':
    # auto_run_params()
    # run_NEZHA()
    run_other()


