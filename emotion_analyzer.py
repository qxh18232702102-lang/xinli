import os
import torch
import numpy as np
import traceback 
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class EmotionAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ===================== Layer 1: 基础情感极性 =====================
        self.model1_dir = "models/erl_sentiment" 
        try:
            self.tokenizer1 = AutoTokenizer.from_pretrained(self.model1_dir, local_files_only=True)
            self.model1 = AutoModelForSequenceClassification.from_pretrained(self.model1_dir, local_files_only=True)
            
            # 🔥 强行挂载中文翻译字典
            self.model1.config.id2label = {"0": "负向", "1": "中性", "2": "正向"}
            self.model1.to(self.device).eval()
            print("✅ Layer 1 原版极性漏斗加载成功！")
        except Exception as e: 
            print(f"❌ Layer 1 加载失败: {e}")
            self.model1 = None

        # ===================== Layer 2: 自主训练模型 =====================
        self.model2_dir = "models/my_finetuned_model"
        print(f"🚀 [系统启动] 正在加载自主训练模型: {self.model2_dir}")
        try:
            self.tokenizer2 = AutoTokenizer.from_pretrained(self.model2_dir, local_files_only=True)
            self.model2 = AutoModelForSequenceClassification.from_pretrained(self.model2_dir, local_files_only=True)
            self.model2.to(self.device).eval()
            print("✅ 自主模型加载成功！")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            self.model2 = None

        self.emotion_labels = ["中性", "难过", "厌恶", "恐惧", "高兴", "惊讶", "愤怒"]

    def _predict_layer(self, text, tokenizer, model):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(self.device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_class = int(np.argmax(probs))
        return pred_class, float(probs[pred_class]), probs

    def analyze(self, text):
        if not text.strip(): return {"最终情绪": "中性", "具体情绪": "中性", "recommendations": []}

        fine = "中性"
        fine_conf = 0.0
        fine_probs_map = {}

        if self.model2:
            try:
                p2, fine_conf, probs2 = self._predict_layer(text, self.tokenizer2, self.model2)
                if p2 < len(self.emotion_labels):
                    current_emotion = self.emotion_labels[p2]
                    fine_probs_map = {self.emotion_labels[i]: float(probs2[i]) for i in range(len(self.emotion_labels))}
                else:
                    current_emotion = "未知"
                fine = current_emotion
            except Exception as e:
                print(f"❌ 推理出错: {e}")

        # 👇 这里的缩进已经完美了，绝不会报错
        direction = "未知"
        if self.model1:
            try:
                p1, _, _ = self._predict_layer(text, self.tokenizer1, self.model1)
                direction = self.model1.config.id2label.get(str(p1), "未知")
                # 🔥 加一行打印，让咱们在终端里看看它到底算出了啥
                print(f"🔍 Layer1 底层结果: 预测数字 p1={p1}, 翻译为={direction}")
            except Exception as e: 
                print(f"❌ Layer 1 推理报错: {e}") 

        return {
            "情感极性": direction,
            "具体情绪": fine,
            "最终情绪": fine,
            "极性置信度": 0.0,
            "情绪置信度": fine_conf,
            "各情绪概率": fine_probs_map
        }