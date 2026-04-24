# PaddlePaddle 兼容层
from paddle_compat import torch

# 位于 data.py 文件的最上方
import numpy as np
import os
import sys
from PIL import Image
from torchvision import transforms
# [新增] 导入 GaussianBlur
from torchvision.transforms import GaussianBlur 
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

class Data:
    def __init__(self, dataset):
        self.dataset = dataset
        self.load_datasets()

        # --- [核心修改] 步骤四: 实现“弱-强对齐”  ---
        # 1. "弱增强" (view_1) 
        self.weak_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 2. "强增强" (view_2) [cite: 23]
        self.strong_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 0.8)),
            transforms.RandomApply([
                GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
            ], p=0.8),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 3. [保留] 测试集/评估集的Tansforms (无随机性)
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
        self.weak_transform  = weak_transform
        self.strong_transform = strong_transform
        self.dataset = dataset
        # self.root = './data/SCID/' # <-- [修复] 已删除，路径在 .txt 中
            
    def __getitem__(self, index):
        # [修复] self.data[index] 已经是正确的相对路径 (例如 'data/SCID/jpg/SCI15.jpg')
        #
        file_path = self.data[index]
        
        try:
            pilImg = Image.open(file_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Cannot find file at {file_path}")
            # 返回一个虚拟数据以避免训练崩溃
            return torch.randn(3, 224, 224), torch.randn(3, 224, 224), index, torch.zeros(40) 
            
        imgi = self.weak_transform(pilImg)
        imgj = self.strong_transform(pilImg)
        
        return (imgi, imgj, index, self.labels[index])

    def __len__(self):
        return len(self.data)

class MyTestDataset(Dataset):
    def __init__(self,data,labels, transform, dataset):
        self.data = data
        self.labels = labels
        self.transform  = transform
        self.dataset = dataset
        # self.root = './data/SCID/' # <-- [修复] 已删除

    def __getitem__(self, index):
        # [修复] self.data[index] 已经是正确的相对路径
        file_path = self.data[index]
        
        try:
            img = Image.open(file_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Cannot find file at {file_path}")
            return torch.randn(3, 224, 224), index, torch.zeros(40)
            
        return (self.transform(img), index, self.labels[index])

    def __len__(self):
        return len(self.data)

def get_scid(): 
    root = './data/SCID/' 
    base_folder = 'train.txt'
    data = []
    labels = []
    
    # [保留] 40 个身份
    num_classes = 40 
    
    filename = os.path.join(root, base_folder)
    
    # --- (开始加载 train.txt) ---
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()
            if not lines:
                break
            
            parts = lines.split()
            if len(parts) < 2: continue
            
            pos_tmp = parts[0]
            # [修复] 删除 os.path.join(root, pos_tmp)
            label_tmp = int(parts[1])
            
            data.append(pos_tmp)
            labels.append(label_tmp)

    data = np.array(data)
    # [保留] 强制 dtype=np.int64
    labels_np = np.array(labels, dtype=np.int64) 
    Y_train = np.eye(num_classes)[labels_np]
    X_train = data
    # --- (结束加载 train.txt) ---
    
    
    base_folder = 'test.txt'
    data = []
    labels = []
    filename = os.path.join(root, base_folder)
    # --- (开始加载 test.txt) ---
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()
            if not lines:
                break

            parts = lines.split()
            if len(parts) < 2: continue
            
            pos_tmp = parts[0]
            # [修复] 删除 os.path.join(root, pos_tmp)
            label_tmp = int(parts[1])
            
            data.append(pos_tmp)
            labels.append(label_tmp)
            
    data = np.array(data)
    # [保留] 强制 dtype=np.int64
    labels_np = np.array(labels,dtype=np.int64)
    Y_test = np.eye(num_classes)[labels_np]
    X_test = data
    # --- (结束加载 test.txt) ---

    X_val = X_test
    Y_val = Y_test

    base_folder = 'database.txt'
    data = []
    labels = []
    filename = os.path.join(root, base_folder)
    # --- (开始加载 database.txt) ---
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()
            if not lines:
                break
            
            parts = lines.split()
            if len(parts) < 2: continue
            
            pos_tmp = parts[0]
            # [修复] 删除 os.path.join(root, pos_tmp)
            label_tmp = int(parts[1])
            
            data.append(pos_tmp)
            labels.append(label_tmp)

    data = np.array(data)
    # [保留] 强制 dtype=np.int64
    labels_np = np.array(labels,dtype=np.int64)
    Y_database = np.eye(num_classes)[labels_np]
    X_database = data
    # --- (结束加载 database.txt) ---

    print("Load SCID dataset complete...") 
    print(f"  Train set:    {len(X_train)} images")
    print(f"  Database set: {len(X_database)} images")
    print(f"  Test set:     {len(X_test)} images")

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, X_database, Y_database

# import numpy as np
# import os
# import sys
# from PIL import Image
# from torchvision import transforms
# # [新增] 导入 GaussianBlur
# from torchvision.transforms import GaussianBlur 
# from torch.utils.data import Dataset, DataLoader
# import torch.multiprocessing

# torch.multiprocessing.set_sharing_strategy('file_system')

# class Data:
#     def __init__(self, dataset):
#         self.dataset = dataset
#         self.load_datasets()

#         # --- [核心修改] 步骤四: 实现“弱-强对齐”  ---
#         # 1. "弱增强" (view_1) 
#         self.weak_transforms = transforms.Compose([
#             transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
#             transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])

#         # 2. "强增强" (view_2)
#         self.strong_transforms = transforms.Compose([
#             transforms.RandomResizedCrop(224, scale=(0.2, 0.8)),
#             transforms.RandomApply([
#                 GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
#             ], p=0.5),
#             transforms.RandomApply([
#                 transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
#             ], p=0.8),
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])

#         # 3. [保留] 测试集/评估集的Tansforms (无随机性)
#         self.test_transforms = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])                                 
#         ])

#     def load_datasets(self):
#         raise NotImplementedError

#     def get_loaders(self, batch_size, num_workers, shuffle_train=False,
#                     get_test=True):
        
#         train_dataset = MyTrainDataset(self.X_train, self.Y_train, 
#                                      self.weak_transforms,
#                                      self.strong_transforms,
#                                      self.dataset)

#         val_dataset = MyTestDataset(self.X_val, self.Y_val, self.test_transforms, self.dataset)
#         test_dataset = MyTestDataset(self.X_test, self.Y_test, self.test_transforms, self.dataset)
#         database_dataset = MyTestDataset(self.X_database, self.Y_database, self.test_transforms, self.dataset)

#         train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
#                                                 shuffle=shuffle_train,
#                                                 num_workers=num_workers)

#         val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
#                                                 shuffle=False,
#                                                 num_workers=num_workers)

#         test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
#                                                 shuffle=False,
#                                                 num_workers=num_workers) if get_test else None

#         database_loader = DataLoader(dataset=database_dataset, batch_size=batch_size,
#                                                     shuffle=False,
#                                                     num_workers=num_workers)
        
#         return train_loader, val_loader, test_loader, database_loader

# class LabeledData(Data):
#     def __init__(self, dataset):
#         super().__init__(dataset=dataset)
    
#     def load_datasets(self):
#         # [修改] 识别 MyScreenDataset
#         if self.dataset == 'my_screen_dataset' or self.dataset == 'scid':  
#             self.topK = 1000  
#             # 调用新的加载函数
#             self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test, self.X_database, self.Y_database = get_myscreendataset()
#         else:
#             raise NotImplementedError(f"Dataset {self.dataset} not supported.")

# class MyTrainDataset(Dataset):
#     def __init__(self, data, labels, weak_transform, strong_transform, dataset):
#         self.data = data
#         self.labels = labels
#         self.weak_transform  = weak_transform
#         self.strong_transform = strong_transform
#         self.dataset = dataset
            
#     def __getitem__(self, index):
#         file_path = self.data[index]
        
#         try:
#             pilImg = Image.open(file_path).convert('RGB')
#         except FileNotFoundError:
#             print(f"Error: Cannot find file at {file_path}")
#             # 返回一个虚拟数据以避免训练崩溃
#             return torch.randn(3, 224, 224), torch.randn(3, 224, 224), index, torch.zeros(1) 
            
#         imgi = self.weak_transform(pilImg)
#         imgj = self.strong_transform(pilImg)
        
#         return (imgi, imgj, index, self.labels[index])

#     def __len__(self):
#         return len(self.data)

# class MyTestDataset(Dataset):
#     def __init__(self,data,labels, transform, dataset):
#         self.data = data
#         self.labels = labels
#         self.transform  = transform
#         self.dataset = dataset

#     def __getitem__(self, index):
#         file_path = self.data[index]
        
#         try:
#             img = Image.open(file_path).convert('RGB')
#         except FileNotFoundError:
#             print(f"Error: Cannot find file at {file_path}")
#             return torch.randn(3, 224, 224), index, torch.zeros(1)
            
#         return (self.transform(img), index, self.labels[index])

#     def __len__(self):
#         return len(self.data)

# def get_myscreendataset(): 
#     # [修改] 指向你的 MyScreenDataset 路径
#     root = './data/MyScreenDataset/' 
    
#     # 1. 加载 Train List
#     base_folder = 'train.txt'
#     data = []
#     labels = []
    
#     filename = os.path.join(root, base_folder)
#     if not os.path.exists(filename):
#         raise FileNotFoundError(f"Train file not found: {filename}. Did you run create_txt_files.py?")

#     with open(filename, 'r') as file_to_read:
#         while True:
#             lines = file_to_read.readline().strip()
#             if not lines:
#                 break
            
#             parts = lines.split()
#             if len(parts) < 2: continue
            
#             pos_tmp = parts[0]
#             # [重要] 恢复路径拼接
#             # txt里存的是 "jpg/img.png"，我们需要它变成 "./data/MyScreenDataset/jpg/img.png"
#             full_path = os.path.join(root, pos_tmp) 
#             label_tmp = int(parts[1])
            
#             data.append(full_path)
#             labels.append(label_tmp)

#     data = np.array(data)
#     labels_np = np.array(labels, dtype=np.int64)
    
#     # [自动计算 num_classes]
#     # 你的数据集有 199 张图，所以 num_classes 应该是 199，而不是 40。
#     # 这里我们通过取标签的最大值+1来动态决定。
#     if len(labels_np) > 0:
#         num_classes = np.max(labels_np) + 1
#     else:
#         num_classes = 0
#         print("Warning: Train set is empty!")

#     print(f"Detected {num_classes} classes from training set.")

#     Y_train = np.eye(num_classes)[labels_np]
#     X_train = data
    
#     # 2. 加载 Test List (Query)
#     base_folder = 'test.txt'
#     data = []
#     labels = []
#     filename = os.path.join(root, base_folder)
    
#     with open(filename, 'r') as file_to_read:
#         while True:
#             lines = file_to_read.readline().strip()
#             if not lines:
#                 break

#             parts = lines.split()
#             if len(parts) < 2: continue
            
#             pos_tmp = parts[0]
#             full_path = os.path.join(root, pos_tmp)
#             label_tmp = int(parts[1])
            
#             data.append(full_path)
#             labels.append(label_tmp)
            
#     data = np.array(data)
#     labels_np = np.array(labels,dtype=np.int64)
#     Y_test = np.eye(num_classes)[labels_np]
#     X_test = data
    
#     # 验证集直接复用测试集
#     X_val = X_test
#     Y_val = Y_test

#     # 3. 加载 Database List (Gallery/Attack Images)
#     base_folder = 'database.txt'
#     data = []
#     labels = []
#     filename = os.path.join(root, base_folder)
    
#     with open(filename, 'r') as file_to_read:
#         while True:
#             lines = file_to_read.readline().strip()
#             if not lines:
#                 break
            
#             parts = lines.split()
#             if len(parts) < 2: continue
            
#             pos_tmp = parts[0]
#             full_path = os.path.join(root, pos_tmp)
#             label_tmp = int(parts[1])
            
#             data.append(full_path)
#             labels.append(label_tmp)

#     data = np.array(data)
#     labels_np = np.array(labels,dtype=np.int64)
#     Y_database = np.eye(num_classes)[labels_np]
#     X_database = data

#     print("Load MyScreenDataset complete...") 
#     print(f"  Train/Query set: {len(X_train)} images")
#     print(f"  Database set:    {len(X_database)} images")
#     print(f"  Num Classes:     {num_classes}")

#     return X_train, Y_train, X_val, Y_val, X_test, Y_test, X_database, Y_database