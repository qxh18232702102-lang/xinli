import json
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ================= 🚀 核心修改区 ================= 
#作用：NLP 文本模型的训练脚本。它读取 train_data.json，配置了 AdamW 优化器、2e-5 的学习率，将底层的 BERT 模型微调成了你的 7 分类专属大脑，并输出了 my_finetuned_model
# 1. 路径换成 BERT
BASE_MODEL_PATH = "models/bert_base" 

# 2. 输出路径不变（这样 analyzer 不用改代码就能读）
OUTPUT_DIR = "models/my_finetuned_model"

# 3. BERT 专用参数 (更稳健的配置)
EPOCHS = 20           # 多练几轮，让它学扎实
BATCH_SIZE = 4
LEARNING_RATE = 2e-5  # BERT 建议用 2e-5 (比之前的 5e-5 更细腻)

# ===============================================

# 强制 ID 映射 (0-6)
LABEL_MAP = {
    0: "中性",
    1: "难过",
    2: "厌恶",
    3: "恐惧",
    4: "高兴",
    5: "惊讶",
    6: "愤怒"
}

class EmotionDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_len=128):
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = int(item['label'])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 [切换引擎] 正在启动 BERT 训练，设备: {device}")
    
    # 加载 BERT 底座
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_PATH, num_labels=7, local_files_only=True
        )
    except Exception as e:
        print(f"❌ 错误：找不到模型文件 {BASE_MODEL_PATH}")
        print("请先运行 download_bert.py 下载模型！")
        return

    model.to(device)
    model.train()

    dataset = EmotionDataset('train_data.json', tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"🔥 BERT 正在努力学习你的数据 (共{EPOCHS}轮)...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print(f"   Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.4f}")

    # 保存结果到 my_finetuned_model
    print(f"💾 训练完成！新大脑已保存到: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # 写入 config 确保标签对齐
    import json as js
    config_path = f"{OUTPUT_DIR}/config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = js.load(f)
    config['id2label'] = {str(k): v for k, v in LABEL_MAP.items()}
    config['label2id'] = {v: k for k, v in LABEL_MAP.items()}
    with open(config_path, 'w', encoding='utf-8') as f:
        js.dump(config, f, indent=4, ensure_ascii=False)

    print("✅ 换脑成功！请重启 app.py 进行测试。")

if __name__ == '__main__':
    train()