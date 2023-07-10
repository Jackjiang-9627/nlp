import os

from flask import Flask, request, jsonify

from predict import Predictor


ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)
predictor = Predictor(
    ckpt_path=os.path.abspath(os.path.join(ROOT_DIR, "datas/output/v1/models/model_000100.pkl")),
    token_vocab_file=os.path.abspath(os.path.join(ROOT_DIR, "datas/output/vocab.pkl")),
    label_vocab_file=os.path.abspath(os.path.join(ROOT_DIR, "datas/output/label_vocab.pkl")),
)
"""用postman测试服务接口是否正常
post：request.form  Body中选择from-data，输入key-value对
get：request.args   Params中输入key-value对
"""

@app.route('/')
def index():
    return "欢迎使用Flask搭建对话意图识别服务器端!"


@app.route("/classify", methods=['GET', 'POST'])
def classify():
    # 获取当前参数字典
    args = request.form if request.method == 'POST' else request.args

    # 获取请求参数
    text = args.get('text', '')
    # 判断参数是否异常
    if not text:
        return jsonify({'code': 201, 'msg': '请给定有效参数:text'})
    # 调用模型获取得到text对应的预测结果信息
    result = predictor.predict(text, k=5)
    # 结果返回
    return jsonify({'code': 200, 'msg': '成功!', 'data': {'text': text, 'intention': result}})


if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=9999
    )