from itertools import chain


def run_BERTology():
    """bert系列"""
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
        # run(ex_index, bert_root_dir, param)

if __name__ == '__main__':
    run_BERTology()