import os.path
from utils import EN_DICT

def get_train_eval_data(data_path, save_path):
    """
    将原始数据处理成：每个词与类别标签  一一对应的格式。 比如：
    文本：我-爱-中-国。
    类别：0-1-2-3-5
    :param data_path: 原始数据的路径
    :param save_path: 数据处理后的保存路径
    :return: 在'./data/sentence_tag'目录下生成  train_txt 和 test.txt文件
    """
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(data_path, 'r', encoding='utf-8') as reader:
        texts = []
        tags = []
        for line in reader:
            line = eval(line.strip())
            # list将字一个个切分开
            text = list(line['originalText'].replace('\r\n', '🚗').replace(' ', '🚗'))
            tag = ['O' for _ in text]
            entities = line['entities']
            for entity in entities:
                # 获取当前实体类别
                en_type = entity['label_type']
                # 获取当前实体范围
                start_pos = entity['start_pos']
                end_pos = entity['end_pos']
                en_lengths = end_pos - start_pos
                # 获取当前实体标注
                en_label = EN_DICT[en_type]
                # 替换实体标签 tag
                if en_lengths == 1:
                    tag[start_pos] = f'S-{en_label}'
                else:
                    tag[start_pos] = f'B-{en_label}'
                    tag[start_pos+1:end_pos] = [f'M-{en_label}' for _ in range(en_lengths - 1)]
                    tag[end_pos-1] = f'E-{en_label}'
                assert len(text) == len(tag), '生成的text与tag长度大小必须一致！'
            tags.append(tag)
            texts.append(text)

    with open(save_path, 'w', encoding='utf-8') as write:
        for text, tag in zip(texts, tags):
            write.writelines(f"{' '.join(text)}\n")
            write.writelines(f"{' '.join(tag)}\n")


if __name__ == '__main__':
    get_train_eval_data(
        data_path='./datas/original_data/training.txt',
        save_path='./datas/sentence_tag/train.txt'
    )
    get_train_eval_data(
        data_path='./datas/original_data/test.json',
        save_path='./datas/sentence_tag/val.txt'
    )
