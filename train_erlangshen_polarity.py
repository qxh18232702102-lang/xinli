import json
import torch
import os
import logging
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ================= 🚀 配置区 =================
BASE_MODEL_NAME = "IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment"
DATA_FILE = "train_data.json"
OUTPUT_DIR = "models/my_erlangshen_2class"
TRACE_DIR = "models/training_traces" # 📁 所有证据图表都会保存在这里

EPOCHS = 10           
BATCH_SIZE = 8        
LEARNING_RATE = 2e-5  

# 动态降维映射：7分类 -> 2分类 (0:负向, 1:正向)
POLARITY_MAP = {0: 1, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 0}
TARGET_LABELS = {0: "负向 (Negative)", 1: "正向 (Positive)"}

# ================= 📝 日志系统配置 =================
os.makedirs(TRACE_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{TRACE_DIR}/training.log", encoding='utf-8'),
        logging.StreamHandler() # 同时输出到控制台和文件
    ]
)
# 过滤掉 transformers 烦人的警告
logging.getLogger("transformers").setLevel(logging.ERROR)

class PolarityDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_len=128):
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        original_label = int(self.data[idx]['label'])
        polarity_label = POLARITY_MAP[original_label]
        
        encoding = self.tokenizer(
            text, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True,
            return_attention_mask=True, return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(polarity_label, dtype=torch.long)
        }

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"🚀 [极性微调启动] 正在使用 {device} 算力引擎...")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True
    )
    model.to(device)

    # 1. 划分训练集和验证集 (80% 训练, 20% 验证) 这样才能画出真实的曲线
    full_dataset = PolarityDataset(DATA_FILE, tokenizer)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # 用于记录画图的数据
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    logging.info(f"🔥 开始进行领域极性微调 (共{EPOCHS}轮, 训练集:{train_size}条, 验证集:{val_size}条)...")
    
    for epoch in range(EPOCHS):
        # -- 训练阶段 --
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        avg_train_loss = total_train_loss / len(train_loader)

        # -- 验证阶段 --
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += outputs.loss.item()
                
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct / total

        # 记录数据
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        logging.info(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # ================= 🎨 绘制证据图表 =================
    logging.info("📊 正在生成训练痕迹图表...")
    
    # 设置中文字体防止乱码
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 绘制 Loss & Accuracy 曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['val_loss'], label='Val Loss', marker='s')
    plt.title('模型收敛曲线 (Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Validation Accuracy', color='green', marker='^')
    plt.title('模型准确率提升曲线 (Accuracy)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(f"{TRACE_DIR}/training_curves.png", dpi=300)
    plt.close()

    # 2. 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["预测: 负向", "预测: 正向"], 
                yticklabels=["实际: 负向", "实际: 正向"])
    plt.title('验证集混淆矩阵 (Confusion Matrix)')
    plt.savefig(f"{TRACE_DIR}/confusion_matrix.png", dpi=300)
    plt.close()

    # 3. 输出分类评估报告
    report = classification_report(all_labels, all_preds, target_names=["负向", "正向"])
    with open(f"{TRACE_DIR}/classification_report.txt", "w", encoding="utf-8") as f:
        f.write("=== 最终模型分类性能评估报告 ===\n\n")
        f.write(report)

    # ================= 💾 保存模型 =================
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    config_path = os.path.join(OUTPUT_DIR, "config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    config['id2label'] = {str(k): v for k, v in TARGET_LABELS.items()}
    config['label2id'] = {v: k for k, v in TARGET_LABELS.items()}
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    logging.info(f"✅ 训练大功告成！所有证据已保存在 {TRACE_DIR} 文件夹下。")

if __name__ == '__main__':
    train()