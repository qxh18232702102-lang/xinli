import cv2
import os

# ================= 配置区 =================
# 保存路径 (自动存入你的训练集文件夹)
DATA_DIR = 'datasets/Custom_Emotion_DB/train' 

# 标签配置：只保留高兴和惊讶
LABELS = {
    '3': 'happy',    # 按 '3' 录入高兴
    '6': 'surprise'  # 按 '6' 录入惊讶
}
# =======================================

# 1. 自动创建文件夹
for label in LABELS.values():
    os.makedirs(os.path.join(DATA_DIR, label), exist_ok=True)

# 2. 加载人脸检测器
haar_path = "haarcascade_frontalface_default.xml"
if not os.path.exists(haar_path):
    # 如果当前目录没有，去系统目录找
    haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier(haar_path)

# 3. 打开摄像头
cap = cv2.VideoCapture(0)

print("="*50)
print("📸 [高兴 & 惊讶] 专用采集器启动！")
print("--------------------------------------------------")
print("😄 录入 [高兴] -> 请按住键盘 '3' 不放")
print("😲 录入 [惊讶] -> 请按住键盘 '6' 不放")
print("")
print("按 'q' -> 退出程序")
print("="*50)

count = 0

while True:
    ret, frame = cap.read()
    if not ret: break

    # 镜像翻转，看起来像照镜子
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 检测人脸 (参数调优，更灵敏)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    # 绘制界面提示
    for (x, y, w, h) in faces:
        # 画绿框
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # 头顶文字
        cv2.putText(frame, "Target Locked", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # 屏幕左上角显示操作提示
    cv2.putText(frame, "Press '3' for Happy", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "Press '6' for Surprise", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow('Data Collector (Happy & Surprise)', frame)
    
    # 监听键盘
    key = cv2.waitKey(1) & 0xFF
    char_key = chr(key)

    # 如果按下了功能键
    if char_key in LABELS:
        folder_name = LABELS[char_key]
        
        if len(faces) > 0:
            # 找到最大的人脸
            (x, y, w, h) = max(faces, key=lambda b: b[2] * b[3])
            
            # 裁剪 + 缩放 (48x48)
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (48, 48))
            
            # 保存文件 (文件名加前缀防止覆盖)
            filename = f"user_{folder_name}_{count}.jpg"
            save_path = os.path.join(DATA_DIR, folder_name, filename)
            
            cv2.imwrite(save_path, face_img)
            print(f"✅ 已保存 [{folder_name}]: {filename}")
            count += 1
        else:
            print("⚠️ 未检测到人脸，无法保存！")

    if char_key == 'q':
        break

cap.release()
cv2.destroyAllWindows()