import os
import pandas as pd
import pickle
from tqdm import tqdm

# ================= 配置区域 =================
# 数据集根目录
DATASET_ROOT = './metadata'
# 图片存放目录 (根据你的 tree 结构)
IMG_DIR = os.path.join(DATASET_ROOT, 'DATASET')
# 标签文件
TRAIN_LABELS_FILE = os.path.join(DATASET_ROOT, 'train_labels.csv')
TEST_LABELS_FILE = os.path.join(DATASET_ROOT, 'test_labels.csv')

# 输出文件 (预处理后保存的文件)
OUTPUT_TRAIN_PKL = './dataset/rafdb_train.pkl'
OUTPUT_TEST_PKL = './dataset/rafdb_test.pkl'
# ===========================================

def load_and_process_data(labels_file, split_name):
    print(f"正在处理 {split_name} 数据集...")
    
    # 1. 读取 CSV
    # 假设 csv 没有表头，或者第一列是图片名，第二列是标签
    # 如果你的 csv 有表头，请添加 header=0
    df = pd.read_csv(labels_file, header=None, names=['img_name', 'label'])
    
    data_list = []
    valid_count = 0
    missing_count = 0
    
    # 2. 遍历每一行
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
        img_name = row['img_name'].strip()
        label = int(row['label']) # 确保标签是整数 (1-7)
        
        # 根据你的目录结构构建路径: datasets/DATASET/train/{label}/img_name
        # 注意：你的目录是 train/1, train/2... 对应标签 1, 2...
        img_path = os.path.join(IMG_DIR, split_name, str(label), img_name)
        
        # 3. 检查文件是否存在 (数据清洗)
        if os.path.exists(img_path):
            # 保存格式: (图片绝对路径, 标签)
            # 标签通常映射为 0-6 (PyTorch习惯)，所以这里减 1
            data_list.append((img_path, label - 1)) 
            valid_count += 1
        else:
            missing_count += 1
            # print(f"警告: 找不到图片 {img_path}")

    print(f"✅ {split_name} 处理完成:")
    print(f"   - 有效图片: {valid_count}")
    print(f"   - 缺失图片: {missing_count}")
    
    return data_list

def main():
    # 处理训练集
    train_data = load_and_process_data(TRAIN_LABELS_FILE, 'train')
    
    # 处理测试集
    test_data = load_and_process_data(TEST_LABELS_FILE, 'test')
    
    # 保存为 pickle 文件
    with open(OUTPUT_TRAIN_PKL, 'wb') as f:
        pickle.dump(train_data, f)
    print(f"\n💾 训练集数据已保存至: {OUTPUT_TRAIN_PKL}")
    
    with open(OUTPUT_TEST_PKL, 'wb') as f:
        pickle.dump(test_data, f)
    print(f"💾 测试集数据已保存至: {OUTPUT_TEST_PKL}")
    
    print("\n🎉 预处理全部完成！现在你可以编写 PyTorch Dataset 类来加载这些 .pkl 文件了。")

if __name__ == '__main__':
    main()