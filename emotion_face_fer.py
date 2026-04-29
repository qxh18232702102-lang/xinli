import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import os

# ================= 1. 升级版网络结构 (必须与 train_face_pro.py 完全一致) =================
class FaceCNN(nn.Module):
    def __init__(self):
        super(FaceCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 48->24
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 24->12
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 12->6
            # 🔥 Block 4 (新增)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 6->3
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 512), # 输入维度变为 2304
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 7) 
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# ================= 2. 推理类 (含中心裁剪保底) =================
class FaceEmotionRecognizer:
    def __init__(self, model_path="models/EmoNet_V1.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
        # 加载检测器
        haar_path = "haarcascade_frontalface_default.xml"
        try:
            if os.path.exists(haar_path):
                self.face_cascade = cv2.CascadeClassifier(haar_path)
            else:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            print("✅ 人脸检测模块就绪")
        except:
            self.face_cascade = None

        # 加载模型
        self.model = FaceCNN().to(self.device)
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                print(f"✅ [Pro Max] 4层深度模型加载成功: {model_path}")
            except Exception as e:
                print(f"❌ 模型结构不匹配，请重新运行训练脚本! 错误: {e}")
        
        self.labels = {
            0: "愤怒", 1: "厌恶", 2: "恐惧", 3: "高兴", 
            4: "中性", 5: "难过", 6: "惊讶"
        }

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def predict_emotion(self, img_path):
        if self.model is None: return "模型未就绪", {}

        try:
            cv_img = cv2.imread(img_path)
            if cv_img is None: return "读图失败", {}
            
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            face_img = None

            # 策略 A: OpenCV 抓取
            if self.face_cascade:
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.05, 
                    minNeighbors=3,
                    minSize=(30, 30)
                )
                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                    padding = 20
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(cv_img.shape[1] - x, w + 2*padding)
                    h = min(cv_img.shape[0] - y, h + 2*padding)
                    face_img = cv_img[y:y+h, x:x+w]
                    print(f"✂️ [Pro] 智能抓取: {w}x{h}")

            # 策略 B: 中心裁剪保底
            if face_img is None:
                h, w = cv_img.shape[:2]
                print(f"⚠️ 启用中心裁剪保底")
                center_x, center_y = w // 2, h // 2
                crop_w, crop_h = int(w * 0.6), int(h * 0.6)
                x1 = max(0, center_x - crop_w // 2)
                y1 = max(0, center_y - crop_h // 2)
                x2 = min(w, center_x + crop_w // 2)
                y2 = min(h, center_y + crop_h // 2)
                face_img = cv_img[y1:y2, x1:x2]

            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            image_tensor = self.transform(face_rgb).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                
            pred_label = self.labels[pred_idx]
            
            prob_dict = {}
            probs_np = probs.cpu().numpy()[0]
            for i, label in self.labels.items():
                prob_dict[label] = round(float(probs_np[i]) * 100, 2)

            return pred_label, prob_dict

        except Exception as e:
            print(f"出错: {e}")
            return "错误", {}