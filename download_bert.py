# 文件名: download_bert.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Google 官方中文 BERT
model_name = "bert-base-chinese"
save_directory = "models/bert_base"

print(f"⏳ 正在下载更强的大脑: {model_name} ...")

# 下载并保存 (指定 num_labels=7，提前为它装好7个分类的插槽)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7)

tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print(f"✅ 下载完成！新底座已保存在: {save_directory}")