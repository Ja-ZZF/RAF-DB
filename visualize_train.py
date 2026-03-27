import pickle
import matplotlib.pyplot as plt
import os

# ================= 配置区域 =================
HISTORY_FILE = './output/rafdb_history.pkl'  # 训练脚本保存的历史文件
SAVE_IMAGE_PATH = './output/visualize_train.png' # 输出的图片路径
# ===========================================

def plot_training_history():
    # 1. 检查文件是否存在
    if not os.path.exists(HISTORY_FILE):
        print(f"❌ 错误: 找不到历史文件 '{HISTORY_FILE}'。请先运行训练脚本。")
        return

    # 2. 加载数据
    print(f"📂 正在加载历史数据...")
    with open(HISTORY_FILE, 'rb') as f:
        history = pickle.load(f)
    
    train_loss = history['train_loss']
    train_acc = history['train_acc']
    test_loss = history['test_loss']
    test_acc = history['test_acc']
    
    epochs = range(1, len(train_loss) + 1)

    # 3. 设置绘图风格
    plt.style.use('seaborn-v0_8-darkgrid') # 使用 seaborn 风格，看起来更现代
    # 如果上面的风格报错，可以使用 plt.style.use('ggplot')

    # 创建画布，设置大小 (宽, 高)，分辨率
    plt.figure(figsize=(16, 6))

    # ================= 绘制损失曲线 =================
    plt.subplot(1, 2, 1) # 1行2列，第1个图
    plt.plot(epochs, train_loss, label='Training Loss', color='#1f77b4', linestyle='-', marker='o', markersize=4)
    plt.plot(epochs, test_loss, label='Validation Loss', color='#d62728', linestyle='-', marker='s', markersize=4)
    
    plt.title('Training & Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)

    # ================= 绘制准确率曲线 =================
    plt.subplot(1, 2, 2) # 1行2列，第2个图
    plt.plot(epochs, train_acc, label='Training Accuracy', color='#2ca02c', linestyle='-', marker='o', markersize=4)
    plt.plot(epochs, test_acc, label='Validation Accuracy', color='#ff7f0e', linestyle='-', marker='s', markersize=4)
    
    plt.title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 添加总标题
    plt.suptitle('RAF-DB ResNet50 Training Performance', fontsize=16, y=1.02)

    # 4. 调整布局并保存
    plt.tight_layout()
    plt.savefig(SAVE_IMAGE_PATH, dpi=300) # 保存为300 DPI的高清图
    print(f"✅ 图片已成功保存至: {SAVE_IMAGE_PATH}")
    
    # 显示图片 (如果在服务器上运行，这行代码可能不会弹窗，但savefig依然有效)
    # plt.show()

if __name__ == '__main__':
    plot_training_history()