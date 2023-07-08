# 中文命名实体识别项目

文件说明：
原始数据可以上阿里云申请
先运行preprocess.py，会生成带标签的数据，序列标注为BMOES格式文件说明：
原始数据可以上阿里云申请
先运行preprocess.py，会生成带标签的数据，序列标注为BMOES格式先看main.py，里面有运行的过程以及调优的参数配置
按照main.py中的run函数中的流程一步一步到对应的函数中去看

utils.py: 特殊用途的函数
dataloader: 构造数据集
load_pretrainmodel.py：下载预训练语言模型
model文件夹：构建CNER模型，包括三部分，通过model.py连接
optimization.py：优化器相关、学习率设置
train.py: 训练代码
evaluate.py: 评估代码
metrics.py：指标代码

运行结果的模型太大了，就不上传了
