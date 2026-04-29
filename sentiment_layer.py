import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentLayer:
    def __init__(self, model_dir="models/roberta_sentiment"):
        model_name = "IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment"  # ✅ 推荐模型
        if not os.path.exists(model_dir):
            print(f"🔄 下载并缓存模型：{model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizer.save_pretrained(model_dir)
            model.save_pretrained(model_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

        # 模型标签通常为：0=消极, 1=中性, 2=积极
        self.id2label = {0: "消极", 1: "中性", 2: "积极"}

    def predict(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[0]
            label_id = torch.argmax(probs).item()
            confidence = probs[label_id].item()
        label = self.id2label.get(label_id, "未知")
        return label, confidence
