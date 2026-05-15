# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
import sys

# 设置标准输出编码为 UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ================= 配置区 =================
DATA_DIR = 'datasets/Custom_Emotion_DB'
MODEL_PATH = 'models/EmoNet_V1.pth'
OUTPUT_DIR = 'models/face_evaluation'
BATCH_SIZE = 64

# 7种表情类别（根据数据集结构自动识别）
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ================= 1. 定义网络结构（必须与训练时一致）=================
class FaceCNN(nn.Module):
    def __init__(self):
        super(FaceCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # Block 1: 基础特征 (边缘、线条)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 48 -> 24
            
            # Block 2: 五官特征 (眼睛、嘴巴)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 24 -> 12
            
            # Block 3: 组合特征 (皱眉、咧嘴)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 12 -> 6
            
            # Block 4: 高级语义特征 (微表情判定)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 6 -> 3
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 7) 
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[启动] 人脸模型评估... 设备: {device}")
    
    # ================= 2. 加载模型 =================
    if not os.path.exists(MODEL_PATH):
        print(f"[错误] 找不到模型文件: {MODEL_PATH}")
        return
    
    model = FaceCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"[成功] 模型加载成功: {MODEL_PATH}")
    
    # ================= 3. 准备验证集 =================
    val_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    val_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, 'val'),
        transform=val_transforms
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 获取实际的类别名称（按字母顺序）
    class_names = val_dataset.classes
    print(f"[数据] 验证集就绪: {len(val_dataset)} 张图片")
    print(f"[类别] 类别顺序: {class_names}")
    
    # ================= 4. 在验证集上进行预测 =================
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    print("[预测] 正在进行预测...")
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"[结果] 验证集准确率: {accuracy:.2f}%")
    
    # ================= 5. 生成混淆矩阵 =================
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 绘制混淆矩阵（带数值）
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names,
                cbar_kws={'label': '样本数量'})
    plt.title(f'人脸情绪识别混淆矩阵\n(验证集准确率: {accuracy:.2f}%)', fontsize=14, pad=20)
    plt.xlabel('预测类别', fontsize=12)
    plt.ylabel('真实类别', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    print(f"[保存] 混淆矩阵已保存: {OUTPUT_DIR}/confusion_matrix.png")
    plt.close()
    
    # 绘制归一化混淆矩阵（百分比）
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': '比例'})
    plt.title(f'人脸情绪识别混淆矩阵（归一化）\n(验证集准确率: {accuracy:.2f}%)', fontsize=14, pad=20)
    plt.xlabel('预测类别', fontsize=12)
    plt.ylabel('真实类别', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_normalized.png", dpi=300, bbox_inches='tight')
    print(f"[保存] 归一化混淆矩阵已保存: {OUTPUT_DIR}/confusion_matrix_normalized.png")
    plt.close()
    
    # ================= 6. 计算每个类别的准确率 =================
    class_accuracy = []
    for i in range(len(class_names)):
        class_correct = cm[i, i]
        class_total = cm[i, :].sum()
        class_acc = 100 * class_correct / class_total if class_total > 0 else 0
        class_accuracy.append(class_acc)
        print(f"  {class_names[i]:10s}: {class_acc:.2f}% ({class_correct}/{class_total})")
    
    # 绘制各类别准确率对比图
    plt.figure(figsize=(10, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']
    bars = plt.bar(class_names, class_accuracy, color=colors, alpha=0.8, edgecolor='black')
    
    # 在柱状图上添加数值标签
    for bar, acc in zip(bars, class_accuracy):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.title('各表情类别识别准确率对比', fontsize=14, pad=20)
    plt.xlabel('表情类别', fontsize=12)
    plt.ylabel('准确率 (%)', fontsize=12)
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/class_accuracy_comparison.png", dpi=300, bbox_inches='tight')
    print(f"[保存] 类别准确率对比图已保存: {OUTPUT_DIR}/class_accuracy_comparison.png")
    plt.close()
    
    # ================= 7. 生成分类报告 =================
    report = classification_report(all_labels, all_preds, 
                                   target_names=class_names,
                                   digits=4)
    
    report_path = f"{OUTPUT_DIR}/classification_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("人脸情绪识别模型性能评估报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"模型路径: {MODEL_PATH}\n")
        f.write(f"验证集大小: {len(val_dataset)} 张图片\n")
        f.write(f"总体准确率: {accuracy:.2f}%\n\n")
        f.write("=" * 60 + "\n")
        f.write("各类别详细指标:\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
        f.write("\n" + "=" * 60 + "\n")
        f.write("各类别准确率:\n")
        f.write("=" * 60 + "\n")
        for name, acc in zip(class_names, class_accuracy):
            f.write(f"{name:10s}: {acc:.2f}%\n")
    
    print(f"[保存] 分类报告已保存: {report_path}")
    print(f"\n[完成] 评估完成！所有结果已保存到 {OUTPUT_DIR} 目录")

if __name__ == '__main__':
    main()
