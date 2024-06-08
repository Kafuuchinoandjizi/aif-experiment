import os
import shutil
import random


def split_dataset(dataset_dir, train_dir, val_dir, train_ratio=0.7):
    # 确保输出目录存在
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 获取所有图像文件
    image_files = [f for f in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, f))]

    # 按类别分类图像文件
    cat_images = [f for f in image_files if f.startswith('cat')]
    dog_images = [f for f in image_files if f.startswith('dog')]

    # 按比例划分训练集和验证集
    def split_and_copy(files, train_dir, val_dir, train_ratio):
        random.shuffle(files)
        split_index = int(len(files) * train_ratio)
        train_files = files[:split_index]
        val_files = files[split_index:]

        for f in train_files:
            shutil.copy(os.path.join(dataset_dir, f), os.path.join(train_dir, f))

        for f in val_files:
            shutil.copy(os.path.join(dataset_dir, f), os.path.join(val_dir, f))

    split_and_copy(cat_images, train_dir, val_dir, train_ratio)
    split_and_copy(dog_images, train_dir, val_dir, train_ratio)

def _2class():
    source_folder = "D:/10439/文档/Pycharm projects/aif实验/dataset/val"
    # 定义目标文件夹路径

    target_folder_cat = "D:/10439/文档/Pycharm projects/aif实验/dataset/cat"
    target_folder_dog = "D:/10439/文档/Pycharm projects/aif实验/dataset/dog"
    for filename in os.listdir(source_folder):
        if filename.startswith("cat"):
            # 如果文件名以 "cat" 开头，则将其移动到 cat 文件夹中
            shutil.move(os.path.join(source_folder, filename), os.path.join(target_folder_cat, filename))
        elif filename.startswith("dog"):
            # 如果文件名以 "dog" 开头，则将其移动到 dog 文件夹中
            shutil.move(os.path.join(source_folder, filename), os.path.join(target_folder_dog, filename))



# 定义数据集路径
dataset_dir = r"D:\10439\个人\stduy\大三\大三下\人工智能基础\source_dataset"
train_dir = r"D:\10439\个人\stduy\大三\大三下\人工智能基础\dataset\train"
val_dir = r"D:\10439\个人\stduy\大三\大三下\人工智能基础\dataset\val"

# 调用函数进行数据集划分
split_dataset(dataset_dir, train_dir, val_dir)

if __name__ == '__main__':
    _2class()