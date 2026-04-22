import os
import random

# --- 1. 配置 ---

# 1.1 定义数据集根目录
# (假设此脚本放在 BRCD-main/ 目录下)
DATA_ROOT = '../data/MyScreenDataset/'

# 1.2 您的原始图像所在的文件夹
# (请确保您的图片都在这个子文件夹中)
IMAGE_DIR_NAME = 'images'

# 1.3 类别名称到 ID 的映射
# (根据您上传的图片自动生成)
LABEL_MAP = {
    'OS': 0,
    'EDU': 1,
    'WEB': 2,
    'VIR': 3,
    'DAT': 4,
    'MED': 5,
    'ERR': 6,
    'MOB': 7,
    'SPC': 8,
    'APP': 9,
}

# 1.4 数据集划分比例
TRAIN_RATIO = 0.7
DATABASE_RATIO = 0.15


# TEST_RATIO 会自动设为 (1.0 - 0.7 - 0.15) = 0.15

# --- 2. 脚本主逻辑 ---

def create_index_files():
    print("开始生成数据集索引...")

    image_dir_path = os.path.join(DATA_ROOT, IMAGE_DIR_NAME)
    if not os.path.exists(image_dir_path):
        print(f"错误：图像目录未找到! \n请确保您的图片位于: {image_dir_path}")
        return

    all_image_info = []

    # --- 扫描所有图像并分配标签 ---
    print(f"正在扫描 {image_dir_path} ...")
    for filename in os.listdir(image_dir_path):
        # 确保是图片文件
        if not (filename.lower().endswith('.jpg') or filename.lower().endswith('.png')):
            continue

        # 从文件名 (如 "OS_001.jpg") 提取前缀 (如 "OS")
        prefix = filename.split('_')[0].upper()

        # 从映射中查找数字 ID
        label_id = LABEL_MAP.get(prefix)

        if label_id is not None:
            # 构造 data.py 需要的相对路径
            relative_path = os.path.join(IMAGE_DIR_NAME, filename).replace('\\', '/')
            all_image_info.append((relative_path, label_id))
        else:
            print(f"警告：跳过文件 {filename}，未知的类别前缀 '{prefix}'")

    if not all_image_info:
        print("错误：未找到任何有效的图像文件。")
        return

    print(f"成功扫描到 {len(all_image_info)} 张图像。")

    # --- 随机打乱并划分数据集 ---
    random.shuffle(all_image_info)

    total_count = len(all_image_info)
    train_count = int(total_count * TRAIN_RATIO)
    database_count = int(total_count * DATABASE_RATIO)

    train_set = all_image_info[:train_count]
    database_set = all_image_info[train_count: train_count + database_count]
    test_set = all_image_info[train_count + database_count:]  # 剩余的都是测试集

    # --- 写入文件 ---
    sets_to_write = {
        'train.txt': train_set,
        'database.txt': database_set,
        'test.txt': test_set,
    }

    for filename, data_set in sets_to_write.items():
        output_path = os.path.join(DATA_ROOT, filename)
        try:
            with open(output_path, 'w') as f:
                for (path, label) in data_set:
                    f.write(f"{path} {label}\n")
            print(f"成功写入: {output_path} (包含 {len(data_set)} 条目)")
        except Exception as e:
            print(f"写入 {output_path} 失败: {e}")

    print("\n索引文件全部生成完毕。")


if __name__ == "__main__":
    create_index_files()