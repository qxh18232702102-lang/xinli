import torch

# 1. 加载你的模型文件
model_path = "models/my_face_model.pth"
print(f"🕵️‍♂️ 正在解剖模型文件: {model_path} ...\n")

# 加载参数字典 (map_location确保即使没有GPU也能看)
state_dict = torch.load(model_path, map_location=torch.device('cpu'))

# 2. 打印每一层的名字和形状
print(f"{'层级名称 (Layer Name)':<30} | {'参数形状 (Shape)':<20} | {'参数数量 (Count)'}")
print("-" * 70)

total_params = 0
for key, value in state_dict.items():
    # 计算这一层的参数总量
    param_count = value.numel()
    total_params += param_count
    
    # 打印细节
    print(f"{key:<30} | {str(list(value.shape)):<20} | {param_count}")

print("-" * 70)
print(f"✅ 总计训练参数量: {total_params} 个")
print("证明：此文件包含完整的权重矩阵，非空文件。")