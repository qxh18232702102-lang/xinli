# vision_fer_model.py
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class SimpleFERCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleFERCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*12*12, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class FaceEmotionRecognizer:
    def __init__(self, model_path="models/fer_cnn.pth"):
        self.labels = ['愤怒','厌恶','恐惧','高兴','难过','惊讶','中性']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((48,48)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.model = SimpleFERCNN(num_classes=len(self.labels))
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"[FaceEmotionRecognizer] Loaded FER model from {model_path}")
                self.model.to(self.device)
                self.model.eval()
                self.ready = True
            except Exception as e:
                print("[FaceEmotionRecognizer] Failed to load model:", e)
                self.ready = False
        else:
            print("[FaceEmotionRecognizer] No FER model file found at", model_path)
            self.ready = False

    def predict(self, img_path):
        if not self.ready:
            # fallback behaviour
            return "中性", {l:0.0 for l in self.labels}

        img = Image.open(img_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred = self.labels[int(np.argmax(probs))]
        probs_dict = {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}
        return pred, probs_dict
