# ============ utils/data.py - PaddlePaddle 版本 ============
"""
SCID 数据集数据加载模块 - PaddlePaddle 版本
对应原 PyTorch 版本的完整迁移
"""

import numpy as np
import os
import sys
from PIL import Image
import paddle
import paddle.vision.transforms as transforms
import paddle.vision as vision
from paddle.vision.transforms import GaussianBlur as PaddleGaussianBlur
from paddle.io import Dataset, DataLoader


class Data:
    def __init__(self, dataset):
        self.dataset = dataset
        self.load_datasets()

        # 弱增强 (view_1)
        self.weak_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 强增强 (view_2)
        self.strong_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 0.8)),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
            ], p=0.8),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 测试集 Transforms (无随机性)
        self.test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_datasets(self):
        raise NotImplementedError

    def get_loaders(self, batch_size, num_workers, shuffle_train=False,
                    get_test=True):

        train_dataset = MyTrainDataset(self.X_train, self.Y_train,
                                      self.weak_transforms,
                                      self.strong_transforms,
                                      self.dataset)

        val_dataset = MyTestDataset(self.X_val, self.Y_val, self.test_transforms, self.dataset)
        test_dataset = MyTestDataset(self.X_test, self.Y_test, self.test_transforms, self.dataset)
        database_dataset = MyTestDataset(self.X_database, self.Y_database, self.test_transforms, self.dataset)

        # PaddlePaddle DataLoader
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                 shuffle=shuffle_train,
                                 num_workers=num_workers)

        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                               shuffle=False,
                               num_workers=num_workers)

        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers) if get_test else None

        database_loader = DataLoader(dataset=database_dataset, batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)

        return train_loader, val_loader, test_loader, database_loader


class LabeledData(Data):
    def __init__(self, dataset):
        super().__init__(dataset=dataset)

    def load_datasets(self):
        if self.dataset == 'my_screen_dataset' or self.dataset == 'scid':
            self.topK = 1000
            self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test, self.X_database, self.Y_database = get_scid()
        else:
            raise NotImplementedError(f"Dataset {self.dataset} not supported. Only 'scid' is configured.")


class MyTrainDataset(Dataset):
    def __init__(self, data, labels, weak_transform, strong_transform, dataset):
        self.data = data
        self.labels = labels
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.dataset = dataset

    def __getitem__(self, index):
        file_path = self.data[index]

        try:
            pilImg = Image.open(file_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Cannot find file at {file_path}")
            return paddle.randn([3, 224, 224]), paddle.randn([3, 224, 224]), index, paddle.zeros([40])

        imgi = self.weak_transform(pilImg)
        imgj = self.strong_transform(pilImg)

        return (imgi, imgj, index, self.labels[index])

    def __len__(self):
        return len(self.data)


class MyTestDataset(Dataset):
    def __init__(self, data, labels, transform, dataset):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.dataset = dataset

    def __getitem__(self, index):
        file_path = self.data[index]

        try:
            img = Image.open(file_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Cannot find file at {file_path}")
            return paddle.randn([3, 224, 224]), index, paddle.zeros([40])

        return (self.transform(img), index, self.labels[index])

    def __len__(self):
        return len(self.data)


def get_scid():
    root = './data/SCID/'
    base_folder = 'train.txt'
    data = []
    labels = []
    num_classes = 40

    filename = os.path.join(root, base_folder)

    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()
            if not lines:
                break

            parts = lines.split()
            if len(parts) < 2:
                continue

            pos_tmp = parts[0]
            label_tmp = int(parts[1])

            data.append(pos_tmp)
            labels.append(label_tmp)

    data = np.array(data)
    labels_np = np.array(labels, dtype=np.int64)
    Y_train = np.eye(num_classes)[labels_np]
    X_train = data

    base_folder = 'test.txt'
    data = []
    labels = []
    filename = os.path.join(root, base_folder)

    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()
            if not lines:
                break

            parts = lines.split()
            if len(parts) < 2:
                continue

            pos_tmp = parts[0]
            label_tmp = int(parts[1])

            data.append(pos_tmp)
            labels.append(label_tmp)

    data = np.array(data)
    labels_np = np.array(labels, dtype=np.int64)
    Y_test = np.eye(num_classes)[labels_np]
    X_test = data

    X_val = X_test
    Y_val = Y_test

    base_folder = 'database.txt'
    data = []
    labels = []
    filename = os.path.join(root, base_folder)

    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()
            if not lines:
                break

            parts = lines.split()
            if len(parts) < 2:
                continue

            pos_tmp = parts[0]
            label_tmp = int(parts[1])

            data.append(pos_tmp)
            labels.append(label_tmp)

    data = np.array(data)
    labels_np = np.array(labels, dtype=np.int64)
    Y_database = np.eye(num_classes)[labels_np]
    X_database = data

    print("Load SCID dataset complete...")
    print(f"  Train set:    {len(X_train)} images")
    print(f"  Database set: {len(X_database)} images")
    print(f"  Test set:     {len(X_test)} images")

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, X_database, Y_database
