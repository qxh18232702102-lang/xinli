import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# ⚠️ 第一步：设置你的真实路径和模型！
# ==========================================
# 1. 你的训练集和验证集文件夹路径 (请修改为真实路径)
TRAIN_DIR = r'C:\Users\13276\Desktop\emotion_recommendation\datasets\Custom_Emotion_DB\train' 
VAL_DIR = r'C:\Users\13276\Desktop\emotion_recommendation\datasets\Custom_Emotion_DB\val'     

# 2. 导入你写的那个 4 层 CNN 模型类
# 假设你的模型类在 emotion_face_fer.py 里叫 SimpleCNN 或 CNN4Layer
from emotion_face_fer_attention import FaceCNN  
# 自动检测 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 🔬 1. 消融实验配置 
# ==========================================
ablation_experiments = [
    {
        "exp_name": "Exp 1: Baseline",
        "use_data_aug": False,        # 关闭数据增强
        "label_smoothing": 0.0,       # 关闭标签平滑
        "color": "#d62728",           
        "linestyle": "-"
    },
    {
        "exp_name": "Exp 2: + Data Aug",
        "use_data_aug": True,         # 开启数据增强
        "label_smoothing": 0.0,       
        "color": "#1f77b4",           
        "linestyle": "--"
    },
    {
        "exp_name": "Exp 3: + Data Aug & Label Smoothing",
        "use_data_aug": True,         
        "label_smoothing": 0.1,       # 开启标签平滑
        "color": "#2ca02c",           
        "linestyle": "-"
    }
]

# ==========================================
# 🛠️ 2. 工具函数 (数据处理与损失函数)
# ==========================================
def get_transforms(use_data_aug=False):
    if use_data_aug:
        train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1), # 如果你是单通道灰度图
            transforms.Resize((48, 48)),                 # 统一尺寸
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) 
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return train_transform, val_transform

def get_loss_function(smoothing_factor=0.0):
    if smoothing_factor > 0.0:
        return nn.CrossEntropyLoss(label_smoothing=smoothing_factor)
    else:
        return nn.CrossEntropyLoss()

# ==========================================
# ⚙️ 3. 核心训练循环 (真实的 前向/反向 传播)
# ==========================================
def train_one_experiment(config, num_epochs=20):
    print(f"\n🚀 开始执行: {config['exp_name']}")
    print("-" * 40)
    
    # 1. 挂载真实数据集
    train_transform, val_transform = get_transforms(config['use_data_aug'])
    
    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    # 2. 实例化真实模型
    model = FaceCNN().to(device)  # ✅ 只用这一行！

    # 3. 损失函数与优化器 (Kaiming初始化无需特意写在这里，在模型类里 写好即可)
    criterion = get_loss_function(config['label_smoothing'])
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    best_acc = 0.0
    history_val_acc = []
    history_val_loss = []
    
    for epoch in range(num_epochs):
        # ---------- 🔴 真实的训练阶段 (Train) ----------
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()       # 清空梯度
            outputs = model(images)     # 前向传播
            loss = criterion(outputs, labels) # 计算 Loss
            loss.backward()             # 反向传播
            optimizer.step()            # 更新权重
            
            train_loss += loss.item() * images.size(0)
            
        # ---------- 🟢 真实的验证阶段 (Evaluation) ----------
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad(): # 验证阶段不计算梯度
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        # 计算当前 Epoch 的平均 Loss 和准确率
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = 100.0 * correct / total
        
        # 记录画图数据
        history_val_loss.append(epoch_val_loss)
        history_val_acc.append(epoch_val_acc)
        
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            # 注意：保存的名字加了 config['exp_name'] 前缀，绝对不会覆盖你原来的模型！
            # torch.save(model.state_dict(), f"ablation_{config['exp_name'][:5]}.pth")
            
        print(f"Epoch [{epoch+1}/{num_epochs}] - Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
        scheduler.step()
        
    print(f"✅ 实验结束！最高准确率: {best_acc:.2f}%")
    return best_acc, history_val_acc, history_val_loss

# ==========================================
# 📊 4. 自动化绘图函数
# ==========================================
def plot_ablation_results(results_dict, num_epochs):
    epochs = np.arange(1, num_epochs + 1)
    plt.rcParams.update({'font.size': 12})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for config in ablation_experiments:
        name = config['exp_name']
        val_acc_history = results_dict[name]['acc_history']
        val_loss_history = results_dict[name]['loss_history']
        
        lw = 3 if "Label Smoothing" in name else 2 
        
        ax1.plot(epochs, val_acc_history, marker='', linestyle=config['linestyle'], 
                 color=config['color'], linewidth=lw, label=name)
        ax2.plot(epochs, val_loss_history, marker='', linestyle=config['linestyle'], 
                 color=config['color'], linewidth=lw, label=name)

    ax1.set_title('Validation Accuracy Growth', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower right')

    ax2.set_title('Validation Loss Convergence', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('ablation_study_curves_real.png', dpi=300, bbox_inches='tight')
    print("\n📈 折线图已自动生成并保存为: ablation_study_curves_real.png")
    plt.show()

# ==========================================
# 🎯 5. 主程序执行入口
# ==========================================
if __name__ == "__main__":
    TOTAL_EPOCHS = 30 # 设定跑多少轮
    experiment_results = {}
    
    for exp_cfg in ablation_experiments:
        best_acc, acc_hist, loss_hist = train_one_experiment(exp_cfg, num_epochs=TOTAL_EPOCHS)
        
        experiment_results[exp_cfg['exp_name']] = {
            'best_acc': best_acc,
            'acc_history': acc_hist,
            'loss_history': loss_hist
        }
        
    print("\n" + "="*60)
    print("📊 最终消融实验结果报告 (直接复制进论文):")
    print("="*60)
    print("| 实验设置 | 验证集最高准确率 (Val Acc) |")
    print("| :--- | :--- |")
    for name, data in experiment_results.items():
        print(f"| {name} | {data['best_acc']:.2f}% |")
    print("="*60)
    
    plot_ablation_results(experiment_results, TOTAL_EPOCHS)