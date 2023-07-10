from torch import nn
from tqdm import tqdm

from train_and_eval import Trainer, Evaluator, Predictor


def training_model():
    trainer = Trainer(
        token_vocab_file="./datas/output/vocab.pkl",
        label_vocab_file="./datas/output/label_vocab.pkl",
        output_dir="./datas/output/v1"
    )
    cfg = {
        'token_emb_layer': {
            'name': 'TokenEmbeddingModule',
            'args': [len(trainer.token_vocab), 128]
        },
        'seq_feature_extract_layer': {
            'name': 'LSTMSeqFeatureExtractModule',  # 注意：此处的args参数顺序与模型定义的参数顺序有关
            'args': [256, 'all_mean', 2, True]
        },
        'classify_decision_layer': {
            'name': 'MLPModule',
            'args': [len(trainer.label_vocab), [256, 128], 0.1, nn.ReLU(), True]
        }
    }

    trainer.training(
        cfg,
        data_files='./datas/original/train.csv',
        n_epochs=1000,
        batch_size=16,
        eval_ratio=0.1,
        save_interval_epoch=5,
        log_interval_batch=10,
        ckpt_path=None,
        lr=0.001,
        weight_decay=0.01
    )


def eval_model():
    cfg = {
        'token_emb_layer': {
            'name': 'TokenEmbeddingModule',
            'args': [128]
        },
        'seq_feature_extract_layer': {
            'name': 'LSTMSeqFeatureExtractModule',
            'args': [256, 'all_mean', 2, True]
        },
        'classify_decision_layer': {
            'name': 'MLPModule',
            'args': [[256, 128], 0.1, nn.ReLU(), True]
        }
    }
    evaluator = Evaluator(
        cfg,
        token_vocab_file="./datas/output/vocab.pkl",
        label_vocab_file="./datas/output/label_vocab.pkl",
        output_dir="./datas/output/v1"
    )
    evaluator.eval(
        data_files='./datas/original/train.csv',
        batch_size=16,
    )


def run_app():
    from app import app

    app.run(
        host="0.0.0.0",
        port=9999
    )


def predictor_model():
    predictor = Predictor(
        ckpt_path="./datas/output/v1/models/model_000100.pkl",
        token_vocab_file="./datas/output/vocab.pkl",
        label_vocab_file="./datas/output/label_vocab.pkl",
    )

    text = "快帮我看一下农历的蜡月三十和农历正月三十哪天是春节呢"
    # text = "大话西游是什么时候上演的"
    result = predictor.predict(text, k=10, probability_threshold=0.01, is_debug=True)
    print(result)

    with open("./datas/original/test.csv", 'r', encoding='utf-8') as reader:
        with open("./datas/output/v1/eval/test.csv", "w", encoding='utf-8') as writer:
            for line in tqdm(reader):
                line = line.strip()
                result_list = predictor.predict(line, k=3)
                writer.writelines(f"{line},")
                r = result_list[0]
                writer.writelines(f"{r['class_name']},{r['probability']},")
                if len(result_list) >= 2:
                    r = result_list[1]
                    writer.writelines(f"{r['class_name']},{r['probability']},")
                else:
                    writer.writelines(",,")
                if len(result_list) >= 3:
                    r = result_list[2]
                    writer.writelines(f"{r['class_name']},{r['probability']}\n")
                else:
                    writer.writelines(",\n")


if __name__ == '__main__':
    training_model()
    # eval_model()
    # predictor_model()
    # run_app()
