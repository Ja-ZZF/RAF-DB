import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from PIL import Image
from torchvision import transforms

# ================= 配置区域 =================
TEST_PKL = './dataset/rafdb_test.pkl'
MODEL_PATH = './output/best_rafdb_resnet50.pth' # 确保这里指向你训练好的最佳模型
SAVE_CM_PATH = './output/visualize_model.png'  # 混淆矩阵保存路径

BATCH_SIZE = 64
NUM_CLASSES = 7
NUM_WORKERS = 4

# RAF-DB 的类别名称 (顺序通常对应 0-6)
# 0: Surprise, 1: Fear, 2: Disgust, 3: Happiness, 4: Sadness, 5: Anger, 6: Neutral
CLASS_NAMES = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
# ===========================================

# 1. 定义数据集类 (复用训练时的逻辑)
class RAFDBDataset(torch.utils.data.Dataset):
    def __init__(self, pkl_path, transform=None):
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# 2. 定义模型结构 (必须与训练时一致)
import torchvision.models as models
def get_model():
    model = models.resnet50(pretrained=False) # 不需要预训练权重，因为我们要加载自己的
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    return model

# 3. 主测试流程
def main():
    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    # 数据增强 (测试时通常只做 Resize 和 Normalize，不做随机增强)
    test_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据
    print(f"📂 正在加载测试集: {TEST_PKL}")
    if not os.path.exists(TEST_PKL):
        print(f"❌ 错误: 找不到测试文件 {TEST_PKL}")
        return

    test_dataset = RAFDBDataset(TEST_PKL, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 加载模型
    print(f"⚙️ 正在加载模型: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到模型文件 {MODEL_PATH}，请先运行训练脚本。")
        return

    model = get_model()
    
    # 加载权重
    # 注意：如果训练时用了 DataParallel，这里加载可能需要处理 key 的前缀
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
    except RuntimeError:
        # 兼容处理：如果训练是多卡，推理是单卡（或反之），可能需要去除 'module.' 前缀
        state_dict = torch.load(MODEL_PATH)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k 
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print("ℹ️ 已自动处理模型权重前缀兼容性问题。")

    model = model.to(device)
    model.eval() # 设置为评估模式

    all_preds = []
    all_labels = []

    print("🔍 正在进行测试推理...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 转换为 numpy 数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ================= 计算指标 =================
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\n✅ 测试集总准确率: {accuracy * 100:.2f}%")
    
    # 打印详细的分类报告 (Precision, Recall, F1-score)
    print("\n📊 分类详细报告:")
    print("-" * 60)
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4)
    print(report)

    # ================= 绘制混淆矩阵 =================
    print(f"\n🎨 正在绘制混淆矩阵并保存至 {SAVE_CM_PATH}...")
    
    plt.figure(figsize=(10, 8))
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 使用 Seaborn 绘图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix on RAF-DB Test Set', fontsize=14, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # 旋转 x 轴标签以防重叠
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(SAVE_CM_PATH, dpi=300)
    print(f"✅ 混淆矩阵图片已保存。")
    # plt.show() # 如果在本地运行，取消注释这行可以直接弹窗显示

if __name__ == '__main__':
    main()