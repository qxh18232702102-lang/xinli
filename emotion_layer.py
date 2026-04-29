import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class EmotionLayer:
    def __init__(self, model_dir="models/text_emotion_cached"):
        model_name = "Johnson8187/Chinese-Emotion"
        if not os.path.exists(model_dir):
            print(f"🔄 下载并缓存模型：{model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizer.save_pretrained(model_dir)
            model.save_pretrained(model_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

        self.id2label = {
            0: "高兴",
            1: "悲伤",
            2: "愤怒",
            3: "惊讶",
            4: "恐惧",
            5: "厌恶",
            6: "中性"
        }

    def predict(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[0]
            label_id = torch.argmax(probs).item()
            confidence = probs[label_id].item()
        return self.id2label[label_id], confidence
