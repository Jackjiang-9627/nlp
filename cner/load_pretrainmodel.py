"""1、bert模型"""
# from transformers import AutoTokenizer, AutoModelForMaskedLM
# tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
# model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")

"""2、Albert预训练词向量"""
# import torch
# from transformers import BertTokenizer, AlbertModel
# tokenizer = BertTokenizer.from_pretrained("clue/albert_chinese_tiny")
# albert = AlbertModel.from_pretrained("clue/albert_chinese_tiny")

"""3、roberta预训练词向量"""

from transformers import BertTokenizer, AlbertModel, BertModel, RobertaModel, RobertaTokenizer
tokenizer = BertTokenizer.from_pretrained("clue/roberta_chinese_clue_tiny")
albert = BertModel.from_pretrained("clue/roberta_chinese_clue_tiny")
