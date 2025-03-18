import torch
import pickle

# 加载 .pth 文件
data = torch.load("pushed_data.pth", map_location="cpu")

# 保存为 pickle 文件
with open("pushed_data.pkl", "wb") as f:
    pickle.dump(data, f)
