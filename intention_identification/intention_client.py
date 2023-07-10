import json

import requests


def invoke_flask_app():
    url = "http://121.40.96.93:9999/classify"
    params = {
        'text': '下周三五点提醒我购买5月1日上海到北京的车票'
        # 'text': '有点黑，光线不太好'
        # 'text': '看看湖南卫视喜羊羊开始播放没有'
        # 'text': '大话西游是什么时候上演的'
        # 'text': '看看大话西游'
    }
    # response = requests.get(url, params=params)
    response = requests.post(url, data=params)
    if response.status_code == 200:
        result = response.json()
        if result['code'] == 200:
            print("预测成功!!!")
            print(result['data'])
        else:
            print(f"调用模型异常:{result['msg']}")
    else:
        print(f"调用模型请求异常:{response.status_code}")


def invoke_pai_eas():
    from eas_prediction import PredictClient
    from eas_prediction import StringRequest, StringResponse

    # client初始化: 只需要初始化一次
    client = PredictClient('http://1757826125271350.cn-shenzhen.pai-eas.aliyuncs.com', 'nlp_intention_identification')
    client.set_token('NTZiZmY3NWU2ZDlhYjQ4ZDE3YmM4ZGUxYzYxMjc2NjU1NGM5NWFhZA==')
    client.init()

    request = StringRequest('下周三五点提醒我购买5月1日上海到北京的车票')
    resp: StringResponse = client.predict(request)
    result = str(resp.response_data, encoding='utf-8')
    result_json = json.loads(result)
    print(result)
    print(result_json)
    print(type(result_json))



if __name__ == '__main__':
    invoke_flask_app()
    invoke_pai_eas()
