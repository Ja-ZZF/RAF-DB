import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import pickle
from PIL import Image
from tqdm import tqdm
import time

# ================= 配置区域 =================
# 数据文件路径
TRAIN_PKL = './rafdb_train.pkl'
TEST_PKL = './rafdb_test.pkl'

# 训练超参数
BATCH_SIZE = 512       # A100 显存很大，可以设大一点
NUM_EPOCHS = 30       # 训练轮数
LEARNING_RATE = 0.001 # 学习率
NUM_CLASSES = 7       # RAF-DB 是 7 类表情
NUM_WORKERS = 8       # 数据加载线程数

# 模型保存路径
SAVE_PATH = './best_rafdb_resnet50.pth'
# ===========================================

# 1. 定义数据集类
class RAFDBDataset(Dataset):
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

# 2. 定义数据增强
# 训练集增强：随机裁剪、水平翻转等
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),       # 先放大一点
    transforms.RandomCrop((112, 112)),   # 随机裁剪到 112x112 (论文标准)
    transforms.RandomHorizontalFlip(),   # 随机水平翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # 颜色抖动
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet 标准化
])

# 测试集增强：中心裁剪，不做随机操作
test_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. 构建模型 (使用 ResNet50)
def get_model():
    # 加载预训练的 ResNet50
    model = models.resnet50(pretrained=True)
    
    # 修改最后的分类层，从 1000类 改为 7类
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    return model

# 4. 训练函数
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{running_loss/total:.4f}', 'acc': f'{100.*correct/total:.2f}%'})
        
    return running_loss/len(loader), 100.*correct/total

# 5. 测试函数
def test(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss/len(loader), 100.*correct/total

# ================= 主程序 =================
if __name__ == '__main__':
    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")
    if torch.cuda.device_count() > 1:
        print(f"⚡ 检测到 {torch.cuda.device_count()} 张 GPU，启用多卡并行！")

    # 加载数据
    print("📂 正在加载数据...")
    train_dataset = RAFDBDataset(TRAIN_PKL, transform=train_transform)
    test_dataset = RAFDBDataset(TEST_PKL, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    # 初始化模型
    model = get_model()
    
    # 多卡并行 (DataParallel)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # 使用 Adam 优化器，论文中也是用的 Adam
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    # 学习率衰减策略
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc = 0.0
    
    print(f"\n🔥 开始训练 {NUM_EPOCHS} 个 Epoch...")
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        
        # 学习率更新
        scheduler.step()
        
        print(f"结果 -> 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}% | 测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.2f}%")
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            # 如果是多卡，保存 model.module.state_dict()，否则保存 model.state_dict()
            state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict()
            torch.save(state_dict, SAVE_PATH)
            print(f"💾 发现更好的模型 (Acc: {test_acc:.2f}%)，已保存至 {SAVE_PATH}")

    print(f"\n🎉 训练完成！最佳测试准确率: {best_acc:.2f}%")