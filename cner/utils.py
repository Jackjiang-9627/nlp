import json
import logging
import os.path
import re
import shutil
from itertools import chain
from pathlib import Path
from typing import Optional, List

import torch
from torch import nn
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase, BertConfig, PretrainedConfig, RobertaConfig

from optimization import BertAdam

EN_DICT = {
    '疾病和诊断': 'DIS',
    '手术': 'OPE',
    '解剖部位': 'POS',
    '药物': 'MED',
    '影像检查': 'SCR',
    '实验室检验': 'LAB'
}

# 标签类别一共有4*6+1=25中
# chain将[[],[],...,[]]串起来 --> []
"""
[['B-DIS', 'M-DIS', 'E-DIS', 'S-DIS'], ['B-OPE', 'M-OPE', 'E-OPE', 'S-OPE'], 
['B-POS', 'M-POS', 'E-POS', 'S-POS'], ['B-MED', 'M-MED', 'E-MED', 'S-MED'],
['B-SCR', 'M-SCR', 'E-SCR', 'S-SCR'], ['B-LAB', 'M-LAB', 'E-LAB', 'S-LAB']]
"""
TAGS = list(chain(*map(lambda tag: [f'B-{tag}', f'M-{tag}', f'E-{tag}', f'S-{tag}'], EN_DICT.values())))
TAGS.append('O')
"""表示标签开始和结束，用于CRF"""
START_TAG = "<START_TAG>"
END_TAG = "<END_TAG>"
TAGS.extend([START_TAG, END_TAG])

def split_text(text, max_len, split_pat=r'([，。]“？)', greedy=False):
    """文本分片
    将超过长度的文本分片成多段满足最大长度要求的最长连续子文本
    约束条件：1）每个子文本最大长度不超过max_len；
             2）所有的子文本的合集要能覆盖原始文本。
             3）每个子文本中如果包含实体，那么实体必须是完整的(可选)
        Arguments:
        text {str} -- 原始文本
        max_len {int} -- 最大长度

    Keyword Arguments:
        split_pat {str or re pattern} -- 分割符模式 (default: {SPLIT_PAT})
        greedy {bool} -- 是否选择贪婪模式 (default: {False})
                         贪婪模式：在满足约束条件下，选择子文本最多的分割方式
                         非贪婪模式：在满足约束条件下，选择冗余度最小且交叉最为均匀的分割方式

    Returns:
        tuple -- 返回子文本列表以及每个子文本在原始文本中对应的起始位置列表

    Examples:
        text = '今夕何夕兮，搴舟中流。今日何日兮，得与王子同舟。蒙羞被好兮，不訾诟耻。
                心几烦而不绝兮，得知王子。山有木兮木有枝，心悦君兮君不知。'
        sub_texts, starts = split_text(text, maxlen=30, greedy=False)
        for sub_text in sub_texts:
            print(sub_text)
        print(starts)
        for start, sub_text in zip(starts, sub_texts):
            if text[start: start + len(sub_text)] != sub_text:
            print('Start indice is wrong!')
            break

    """
    # 文本小于max_len则不分割
    if len(text) <= max_len:
        return [text], [0]
    # 分割字符串
    segs = re.split(split_pat, text)
    # init
    sentences = []
    # 将分割后的段落和分隔符组合
    for i in range(0, len(segs) - 1, 2):
        sentences.append(segs[i] + segs[i + 1])
    if segs[-1]:
        sentences.append(segs[-1])
    n_sentences = len(sentences)
    sent_lens = [len(s) for s in sentences]

    # 所有满足约束条件的最长子片段
    alls = []
    for i in range(n_sentences):
        length = 0
        sub = []
        for j in range(i, n_sentences):
            if length + sent_lens[j] <= max_len or not sub:  # not sub: not [] --> True
                sub.append(j)
                length += sent_lens[j]
            else:
                break
        alls.append(sub)
        # 将最后一个段落加入
        if j == n_sentences - 1:
            if sub[-1] != j:
                alls.append(sub[1:] + [j])
            break

    if len(alls) == 1:
        return [text], [0]

    if greedy:
        # 贪婪模式返回所有子文本
        sub_texts = [''.join([sentences[i] for i in sub]) for sub in alls]
        starts = [0] + [sum(sent_lens[:i]) for i in range(1, len(alls))]
        return sub_texts, starts
    else:
        # 用动态规划求解满足要求的最优子片段集
        DG = {}  # 有向图
        N = len(alls)
        for k in range(N):
            tmplist = list(range(k + 1, min(alls[k][-1] + 1, N)))
            if not tmplist:
                tmplist.append(k + 1)
            DG[k] = tmplist

        routes = {}
        routes[N] = (0, -1)
        for i in range(N - 1, -1, -1):
            templist = []
            for j in DG[i]:
                cross = set(alls[i]) & (set(alls[j]) if j < len(alls) else set())
                w_ij = sum([sent_lens[k] for k in cross]) ** 2  # 第i个节点与第j个节点交叉度
                w_j = routes[j][0]  # 第j个子问题的值
                w_i_ = w_ij + w_j
                templist.append((w_i_, j))
            routes[i] = min(templist)

        sub_texts, starts = [''.join([sentences[i] for i in alls[0]])], [0]
        k = 0
        while True:
            k = routes[k][1]
            sub_texts.append(''.join([sentences[i] for i in alls[k]]))
            starts.append(sum(sent_lens[: alls[k][0]]))
            if k == N - 1:
                break

    return sub_texts, starts


class InputExamples(object):
    """一条样本，包含这条样本对应的token set以及tag set"""

    def __init__(self, sentence: List, tag: List):
        super(InputExamples, self).__init__()
        self.sentence = sentence
        self.tag = tag


def read_examples(data_dir: Path, data_sign):
    """将每个样本转化为InputExample对象，最后用list储存起来"""
    examples = []
    with open(data_dir / f'{data_sign}.txt', 'r', encoding='utf-8') as reader:
        for line in reader:
            sentence = line.strip().split(' ')
            if sentence:
                tag = reader.readline().strip().split(' ')
                examples.append(InputExamples(sentence, tag))
            else:  # sentence为空白行
                break
    # print(f'InputExamples:{len(examples)}')
    return examples


class InputFeature(object):
    def __init__(self, input_ids, label_ids, input_mask):
        """
        一条输入特征样本
        :param example_id:
        :param input_ids:
        :param label_ids:
        :param input_mask:
        :param split_to_original_id:
        """
        super(InputFeature, self).__init__()
        # self.example_id = example_id
        self.input_ids = input_ids
        self.label_ids = label_ids
        self.input_mask = input_mask
        # self.example_id = split_to_original_id


def examples2features(examples: List[InputExamples], tokenizer: PreTrainedTokenizerBase,
                      word_dict, tag2idx, max_seq_len, greedy, pad_sign, pad_token):
    """
    将样本转换为特征向量
    :param greedy: 原始样本的文本分割函数是否贪心
    :param pad_token: '[PAD]'
    :param pad_sign: 是否填充
    :param examples: InputExamples的列表
    :param tokenizer: 分词器将文本按字处理成token，需要一个vocab.txt文件来映射token to id的关系
    :param word_dict: 词典vocab，可以自己制作，tokenizer中已包括vocab
    :param tag2idx: 标签到id的映射字典，自己构造
    :param max_seq_len:最大序列长度  用的BertConfig中的 max_position_embeddings=512
    :return: [InputFeature]
    """
    features = []
    split_pat = r'([,.!?，。！？]”?)'
    # tokenizer.tokenize(pad_token) 分词返回一个列表
    pad_token = tokenizer.tokenize(pad_token)[0] if len(tokenizer.tokenize(pad_token)) == 1 else '[UNK]'
    pad_idx = tokenizer.convert_tokens_to_ids(pad_token)
    for (example_idx, example) in tqdm(enumerate(examples), total=len(examples)):
        # 1. 长样本进行split分割, 返回子文本list和开始位置list
        sub_texts, starts = split_text(text=''.join(example.sentence), max_len=max_seq_len,
                                       split_pat=split_pat, greedy=greedy)  # todo: split_text函数还未看懂！

        # 获取每个样本的文本字数
        # original_id = list(range(len(example.sentence)))

        # 获取每个sub_text对应的InputFeature
        for subtext, start in zip(sub_texts, starts):
            # token处理：对子文本分词（单个字），为空即词典不存在，设置为UNK
            # tokenizer.tokenize(token)没有加[0]  --> TypeError: unhashable type: 'list'
            text_tokens = [tokenizer.tokenize(token)[0] if len(tokenizer.tokenize(token)) == 1 else '[UNK]'
                           for token in subtext]
            # token转化为id
            token_ids = tokenizer.convert_tokens_to_ids(text_tokens)
            # 获取对应区域的label id
            label_ids = [tag2idx[tag] for tag in example.tag[start:start + len(subtext)]]
            assert len(token_ids) == len(label_ids), '文本和标签长度不一致'
            # 截断补齐处理
            pad_len = max_seq_len - len(subtext)
            if pad_len < 0:
                token_ids = token_ids[:max_seq_len]
                label_ids = label_ids[:max_seq_len]
                input_mask = [1 for _ in token_ids]
            if pad_sign and pad_len > 0:
                token_ids += [pad_idx] * pad_len
                label_ids += [tag2idx['O']] * pad_len
                # mask
                input_mask = [1 for _ in text_tokens] + [0] * pad_len
            assert len(token_ids) == len(label_ids) == len(input_mask),\
                f'文本{len(token_ids)}、标签{len(label_ids)}、mask{len(input_mask)}长度不一致'
            features.append(InputFeature(
                input_ids=token_ids,
                label_ids=label_ids,
                input_mask=input_mask))
    return features


class Params(object):
    """定义参数对象，参数的定义，可以是外部给定，  也可以在内部直接初始化:Optional[BertConfig]"""

    def __init__(self, config, params=None, ex_index=1):
        super(Params, self).__init__()
        """
        config=None: cfg配置可以直接采用BertConfig对象中的初始化数据,为None
        config=cfg: 自己构造一个配置文件config.json，用BertConfig.from_json_file()导入
        params参数外部以字典形式给定
        ex_index=1: 实验生成文件的id --> run_(ex_index)
        """
        if config is None:
            config = BertConfig()
        self.config: BertConfig = config
        if params is None:
            params = {}
        """1、数据加载器dataloader构造相关参数"""
        self.train_batch_size = 32
        self.val_batch_size = 32
        self.test_batch_size = 128
        # 序列允许的最大长度
        # self.max_seq_len = 128
        self.max_seq_len = config.max_position_embeddings

        self.data_cache = True  # 是否缓存features数据
        # 标签映射
        self.tag2idx = {tag: i for i, tag in enumerate(TAGS)}
        self.num_labels = len(self.tag2idx)

        """2、文件路径参数"""
        # 根目录: 当前utils所在的文件夹   路径如果不给定则采用默认路径
        self.root_dir = Path(params.get('root_dir', os.path.abspath(os.path.dirname(__file__))))

        # 数据集路径  utils所在的文件夹
        self.data_dir = Path(params.get('data_dir', self.root_dir / 'datas'))
        # 分词器 BertTokenizer bert模型的vocab文件路径 一般自行下载后以参数传递
        self.bert_root_dir = Path(params['bert_root_dir'])
        self.bert_vocab_path = self.bert_root_dir / 'vocab.txt'

        # 开始训练后， 生成的文件全部保存在 run文件夹下的  --> run_ex_index
        # 更换模型后需要改变ex_index
        self.run_root_dir = Path(params.get('root_dir', self.root_dir / 'runs' / f"run{ex_index}"))
        # 日志保存路径
        self.params_path = Path(params.get('params_path', self.run_root_dir / 'experiments'))
        self.params_path.mkdir(parents=True, exist_ok=True)
        # 模型保存路径
        self.models_path = Path(params.get('model_path', self.run_root_dir / 'models'))
        self.models_path.mkdir(parents=True, exist_ok=True)

        """3、模型相关参数"""
        # 1、语言模型相关参数，params参数中给定语言模型，默认Word2VecLMModule
        self.lm_layer_name = params.get('lm_layer_name', 'Word2VecLMModule')
        self.lm_freeze_params = False  # True表示冻结language model里面的迁移参数，False表示不冻结
        self.lm_fusion_layers = 4  # 融合后4层数

        # 2、特征提取模型相关参数，params参数中给定encoder模型，默认BiLSTMEncoderModule
        self.encoder_layer_name = params.get('encoder_layer_name', 'BiLSTMEncoderModule')
        self.encoder_output_size = config.hidden_size  # 最终Encoder输出的特征向量维度大小
        self.with_encoder = True
        self.encoder_dropout = 0.3
        # LSTM模型
        self.encoder_lstm_layers = 1  # Encoder BiLSTM的层数
        self.encoder_lstm_hidden_size = config.hidden_size  # 在lstm中，输出E维度就是隐层的size
        self.encoder_lstm_with_ln = True
        # IDCNN
        self.encoder_idcnn_filters = 128  # 卷积核数量，也就是输出通道数量
        # 每层卷积层的参数
        self.encoder_idcnn_conv1d_params = [
            {
                'dilation': 1,  # 膨胀系数
                'kernel_size': 3  # 卷积核大小
            },  # 第一层
            {
                'dilation': 2,
            },  # 第二层
            {
                'dilation': 4,
            },  # 第三层
        ]
        self.encoder_idcnn_kernel_size = 3  # 上面的encoder_idcnn_conv1d_params卷积核大小未给定时采用
        self.encoder_idcnn_num_block = 4  # 卷积块的重复次数（参数共享）

        # RTRANSFORMER

        # 3、分类模型相关参数，params参数中给定分类模型，默认SoftmaxSeqClassifyModule
        self.classify_layer_name = params.get('classify_layer_name', 'SoftmaxSeqClassifyModule')
        self.classify_fc_hidden_size = [512, 256, 128]  # 给定全连接中的神经元数目，可以是None或者int=128或者list[int]=[512, 256, 128]
        self.classify_fc_dropout = 0.0
        # todo：全连接层数， 如果不存在全连接层，encoder输出将会重新赋值为 self.num_labels
        if self.classify_fc_hidden_size is None:
            self.classify_fc_hidden_size = []
        if isinstance(self.classify_fc_hidden_size, int):
            self.classify_fc_hidden_size = [self.classify_fc_hidden_size]
        if len(self.classify_fc_hidden_size) > 0:
            if self.classify_fc_hidden_size[-1] != self.num_labels:
                # 全连接的最后一层不是标签数目大小，直接添加一个大小
                self.classify_fc_hidden_size.append(self.num_labels)
        # 全连接总的层数
        self.classify_fc_layers = len(self.classify_fc_hidden_size)
        if self.classify_fc_layers == 0:
            # 如果classify决策层中不存在全连接，那么encoder输出就是每个token对应每个类别的置信度
            self.encoder_output_size = self.num_labels

        """4、优化器、设备相关参数"""
        # 1、优化器
        self.gradient_accumulation_steps = 1
        # 2、device
        self.multi_gpu = False  # 是否是多GPU运行
        self.n_gpu = 0  # 0表示cpu运行，1表示1个gpu运行，n表示n个gpu运行
        self.gpu_device_id = 0  # 只有一个gpu --> 'cuda0'
        device = torch.device("cpu")
        if torch.cuda.is_available():
            if self.multi_gpu:
                device = torch.device("cuda")  # 多个gpu不分设备
                n_gpu = torch.cuda.device_count()
            else:
                device = torch.device(self.gpu_device_id)  # 单个gpu  --> 'cuda0'
                n_gpu = 1
            self.n_gpu = n_gpu
        self.device = device
        # self.device = torch.device("cpu")

        """5、训练测试相关参数"""
        self.epoch_num = 100
        # lr
        self.lm_tuning_lr = 2e-5
        self.encoder_lr = 1e-4
        self.classify_lr = 1e-4
        self.classify_crf_lr = 0.01
        self.warmup_prop = 0.1  #
        self.warmup_schedule = 'warmup_cosine'
        # weight_decay
        self.lm_weight_decay = 0.01
        self.encoder_weight_decay = 0.01
        self.classify_weight_decay = 0.01
        # 梯度截断
        self.max_grad_norm = 2.0
        # 提前终止训练
        self.stop_epoch_num_threshold = 5  # 最多允许连续3个epoch模型效果不提升 3 --> 5
        self.min_epoch_num_threshold = 5  # 至少要求模型训练5个epoch
        self.stop_improve_val_f1_threshold = 0.0

        """6、参数覆盖,将入参params转化为  self.k = v """
        for k, v in params.items():
            if k in ['n_gpu', 'root_path', 'data_dir', 'params_path', 'model_dir', 'bert_model_root_dir']:
                continue
            self.__dict__[k] = v

    def to_dict(self):
        params = {}
        # noinspection DuplicatedCode
        for k, v in self.__dict__.items():
            if k in ['device']:  # device信息不保存
                continue
            if isinstance(v, Path):  # Path对象转换为绝对路径的字符串
                v = str(v.absolute())
            if isinstance(v, PretrainedConfig):  # BertConfig对象转换为字典
                v = v.to_dict()
            params[k] = v
        return params

    def __str__(self):
        """打印参数，返回字符串表达"""
        params = self.to_dict()
        # json.dumps()是把python对象转换成json对象的一个过程，生成的是字符串。
        # sort_keys = True:是告诉编码器按照字典排序(a到z)输出
        # indent:参数根据数据格式缩进显示，读起来更加清晰，为None显示在一行
        param_str = json.dumps(params, ensure_ascii=False, indent=4)
        return param_str

    @staticmethod
    def load(json_path):
        with open(json_path, 'r', encoding='utf-8') as reader:
            params = json.load(reader)
            cfg = BertConfig.from_dict(params['config'])
            del params['config']
            return Params(config=cfg, ex_index=params['ex_index'], params=params)

    def save_params(self, json_path=None):
        if json_path is None:
            json_path = self.params_path / 'params.json'
        # 将所有参数(包括NEZHAConfig对象的参数，不转会报错)转换为python字典
        params = self.to_dict()
        with open(json_path, 'w', encoding='utf-8') as writer:
            # 将python编码成json放在那个文件里
            json.dump(params, writer, ensure_ascii=False)

def build_params(ex_index, bert_root_dir, input_params):
    # 预训练模型构建所需参数的文件
    bert_root_dir = bert_root_dir
    # lm: --> bert系列模型 [Word2VecLMModule、BertLMModule、ALBertLMModule、RoBertaLMModule]
    # encoder_model: --> [BiLSTMEncoderModule, IDCNNEncoderModule, RTransformerEncoderModule]
    # classify_model: --> [SoftmaxSeqClassifyModule, CRFSeqClassifyModule]
    if input_params['lm_layer_name'] == 'NEZHALMModule':
        from models.lm_models.NEZHA.model_NEZHA import NEZHAConfig
        cfg = NEZHAConfig.from_json_file(os.path.join(os.path.dirname(__file__), bert_root_dir, "config.json"))
    else:
        cfg = BertConfig.from_json_file(os.path.join(os.path.dirname(__file__), bert_root_dir, "config.json"))
    params = Params(
        ex_index=ex_index,
        config=cfg,
        params=input_params
    )

    # nezha模型 [NEZHALMModule] --> 需要减小批次
    # from models.lm_models.NEZHA.model_NEZHA import NEZHAConfig
    # cfg = NEZHAConfig.from_json_file(os.path.join(os.path.dirname(__file__), bert_root_dir, "config.json"))
    # params = Params(ex_index=ex_index,
    #     config=cfg,
    #     params={
    #         'data_dir': r'.\datas\sentence_tag',
    #         'bert_root_dir': bert_root_dir,
    #         'lm_layer_name': 'NEZHALMModule',
    #         'train_batch_size': 16
    #     }
    # )
    return params

class RunningAverage:
    """A simple class that maintains the running average of a quantity
    记录平均损失
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        if self.steps <= 0:
            return 0.0
        else:
            return self.total / float(self.steps)

def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains the entire model, may contain other keys such as epoch, optimizer
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth')
    if not os.path.exists(checkpoint):
        print(f"Checkpoint Directory does not exist! Making directory {checkpoint}")
        os.makedirs(checkpoint)
    torch.save(state, filepath)
    # 如果是最好的checkpoint则以best为文件名保存
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth'))

def set_logger(save=False, log_path=None):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
        save: 是否保存
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    log_path = os.path.abspath(log_path)
    if save and not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))

    logger.handlers = []
    if save:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)


def build_optimizer(model: nn.Module, params: Params, total_train_batch: int):
    """
    优化器构建
    :param model: 待训练的模型
    :param params: 参数对象
    :param total_train_batch: 总的训练批次数量
    :return: 模型优化器
    """
    # 1、获取模型参数
    parameter_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    # 2、参数分组
    lm_parameters = [(n, p) for n, p in parameter_optimizer if n.startswith('lm_layer.')]
    encoder_parameters = [(n, p) for n, p in parameter_optimizer if n.startswith('encoder_layer.')]
    classify_parameters = [(n, p) for n, p in parameter_optimizer if n.startswith('classify_layer.')]
    # 不做权重衰减的参数
    no_decay = ['bias', 'LayerNorm', 'layer_norm', 'dym_weight']
    optimizer_grouped_parameters = [
        # lm_layer + 惩罚系数(L2损失)
        {
            'params': [p for n, p in lm_parameters if not any(nd in n for nd in no_decay)],  # 去掉no_decay
            'weight_decay': params.lm_weight_decay,
            'lr': params.lm_tuning_lr
        },
        # lm_layer
        {
            'params': [p for n, p in lm_parameters if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': params.lm_tuning_lr
        },
        # encoder + 惩罚系数(L2损失)
        {
            'params': [p for n, p in encoder_parameters if not any(nd in n for nd in no_decay)],
            'weight_decay': params.encoder_weight_decay,
            'lr': params.encoder_lr
        },
        # encoder
        {
            'params': [p for n, p in encoder_parameters if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': params.encoder_lr
        },
        # classify_layer + 惩罚系数(L2损失)
        {
            'params': [p for n, p in classify_parameters if (not any(nd in n for nd in no_decay)) and 'crf' not in n],
            'weight_decay': params.classify_weight_decay,
            'lr': params.classify_lr
        },
        # classify_layer
        {
            'params': [p for n, p in classify_parameters if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': params.classify_lr
        },
        # crf参数学习 --> 一般情况下，不加惩罚性，并且学习率比较大
        {
            'params': [p for n, p in classify_parameters if 'crf' in n],
            'weight_decay': 0.0,
            'lr': params.classify_crf_lr
        }
    ]
    # 去除不需要更新的参数
    optimizer_grouped_parameters = [ogp for ogp in optimizer_grouped_parameters if len(ogp['params']) > 0]
    # 3、优化器构建 todo：待看！
    optimizer = BertAdam(
        params=optimizer_grouped_parameters,
        warmup=params.warmup_prop,
        t_total=total_train_batch,  # 给定当前训练中的总的批次数目(可以是近似的，主要影响warmup的执行)
        schedule=params.warmup_schedule,  # 给定warmup学习率变化(前期学习率增大，后期减小)
        max_grad_norm=params.max_grad_norm  # 最大梯度截断
    )
    return optimizer
