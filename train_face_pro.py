import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import time

# ================= 配置区 =================
#作用：面部视觉模型的最终版训练脚本。里面写明了你独创的 4 层 CNN 结构、数据增强（锐化、翻转）以及标签平滑（Label Smoothing）技术
DATA_DIR = 'datasets/Custom_Emotion_DB' 
MODEL_SAVE_PATH = 'models/EmoNet_V1.pth' 
EPOCHS = 40       # 保持 40 轮即可
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# ================= 1. 升级版网络结构 (4层卷积) =================
# ⚠️ 注意：这个结构变了，所以训练完后必须替换 emotion_face_fer.py 里的结构
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
            
            # 🔥 Block 4: 高级语义特征 (新增层：微表情判定)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 6 -> 3
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 512), # 全连接层输入变了
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
    print(f"🚀 启动 [Pro Max] 训练模式... 设备: {device}")

    # ================= 2. 视觉增强 (加入锐化) =================
    train_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3), # 增强对比度对抗逆光
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5), # 🔥 随机锐化 (对付眼镜干扰)
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    val_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if not os.path.exists(DATA_DIR):
        print(f"❌ 找不到数据集: {DATA_DIR}")
        return

    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"📊 数据就绪: 训练集 {len(train_dataset)} | 验证集 {len(val_dataset)}")

    # ================= 3. 训练配置 =================
    model = FaceCNN().to(device)
    
    # 🔥 核心升级: Label Smoothing (防止模型过于自信导致的误判)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0

    # ================= 4. 训练循环 =================
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        scheduler.step()

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"    🌟 新纪录! 模型已保存")

    print(f"\n✅ [Pro Max] 训练完成! 总耗时: {(time.time()-start_time)/60:.2f} 分钟")

    # 绘图
    plt.figure(figsize=(10, 4))
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy Growth (4-Layer CNN + Label Smoothing)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy %')
    plt.legend()
    plt.savefig('models/training_log_promax.png')

if __name__ == '__main__':
    main()