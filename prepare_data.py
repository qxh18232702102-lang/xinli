import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm

# 配置
csv_file = 'fer2013.csv'
output_dir = 'datasets/my_face_data' # 你的私有数据集目录
emotions = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}

def prepare():
    if not os.path.exists(csv_file):
        print("❌ 请先下载 fer2013.csv 放到项目根目录！")
        return

    print("🚀 正在将 CSV 转换为可视化的图片文件夹...")
    df = pd.read_csv(csv_file)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        emotion = emotions[row['emotion']]
        pixels = np.fromstring(row['pixels'], sep=' ')
        image = pixels.reshape(48, 48).astype('uint8')

        # 分类：Training -> train, 其他 -> val
        usage = 'train' if row['Usage'] == 'Training' else 'val'

        # 存图路径: datasets/my_face_data/train/Happy/123.jpg
        save_path = os.path.join(output_dir, usage, emotion)
        os.makedirs(save_path, exist_ok=True)

        cv2.imwrite(f"{save_path}/{idx}.jpg", image)

    print(f"✅ 数据准备完毕！请去 {output_dir} 查看你的数据。")

if __name__ == '__main__':
    prepare()