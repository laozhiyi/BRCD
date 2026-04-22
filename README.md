# BRCD

这是论文 **Bit-mask Robust Contrastive Knowledge Distillation for Unsupervised Semantic Hashing (WWW2024)** (https://arxiv.org/pdf/2403.06071) 的 PyTorch 实现。

### 项目概述

本论文提出了 **Bit-aware Robust Contrastive Knowledge Distillation (BRCD)**，一种专门为无监督语义哈希模型蒸馏而设计的方法，旨在缓解 **ViT** 等大规模骨干网络带来的推理延迟。我们的方法首先通过对比学习目标对齐教师模型和学生模型的语义空间，在个体特征层面和结构语义层面实现知识蒸馏，从而确保语义哈希中两种关键搜索范式的有效性。此外，我们在对比学习目标中融入了基于聚类的策略，以消除噪声增强并确保鲁棒的优化。同时，通过位级分析，我们揭示了由位独立性引起的哈希码冗余问题，并引入了一种 **位掩码机制** 来缓解其影响。大量的实验表明，**BRCD** 在各种语义哈希模型和骨干网络上表现出色，具有强大的泛化能力，显著优于现有的知识蒸馏方法。

![框架图](image/framework_7.png)

### 主要依赖

+ pytorch             1.10.1
+ torchvision         0.11.2
+ numpy               1.19.5
+ pandas              1.1.5
+ Pillow              8.4.0

### 如何运行

我们以使用 **ViT_b_16** 作为教师模型在 CIFAR-10 数据集上运行代码为例进行说明。

首先，运行以下命令来训练教师模型：

```bash
sh scripts/run.sh
```

如果你运行上述命令，程序会将 CIFAR-10 数据集下载到 `./data/cifar10/` 目录，然后开始训练。

然后，使用我们的 BRCD 方法训练学生模型：

```bash
sh scripts/ours_distill_run.sh
```

### 如何运行 (详细命令)

#### 1. 训练教师模型

使用 CIFAR-10 数据集：

```bash
python main.py cifar vit_b_16 --train --dataset cifar10 --encode_length 64 --cuda --device 0 --trail 1 --epochs 200 --lr 0.0005 --num_bad_epochs 3 --batch_size 128 --num_workers 4
```

使用 SCID 数据集：

```bash
python main.py scid vit_b_16 --train --dataset scid --encode_length 64 --cuda --device 0 --trail 1 --epochs 100 --lr 0.0005 --num_bad_epochs 10 --batch_size 128 --num_workers 4
```

#### 2. 训练学生模型 (知识蒸馏)

使用 SCID 数据集进行蒸馏：

```bash
python ours_distill_main.py --s_model_name mobilenet_v2 --t_model_name vit_b_16 --train --dataset scid --encode_length 64 --cuda --device 0 --trail 1 --alpha 0.8 --epochs 400 --lr 0.0005 --num_bad_epochs 10 --batch_size 128 --num_workers 4
```

---

## 项目目录结构详解

```
BRCD-main/
│
├── [根目录文件]
│
├── app.py                    # Web 应用入口，基于 Flask 的图像检索 API 服务
│                              # 启动后在浏览器打开前端页面，可上传图片进行以图搜图
│
├── main.py                   # 训练教师（Teacher）模型的脚本（目前大部分代码已注释）
│                              # 支持 CIFAR-10 / SCID / 自定义数据集
│
├── ours_distill_main.py       # 训练学生（Student）模型的脚本（知识蒸馏主入口）
│                              # 将大型教师模型（ViT）的知识迁移到轻量学生模型（MobileNet V2）
│
├── create_txt_files.py        # 从 gnd.json 自动生成 train.txt / test.txt / database.txt（已注释）
│
├── test_logging.py            # 日志模块测试脚本，验证 Logger 类是否正常工作
│
├── training.log               # 训练过程产生的日志文件
│
├── model/                     # 模型核心定义，所有网络结构和训练逻辑
│   │
│   ├── __init__.py            # Python 包标识文件（使 model 目录可被 import）
│   │
│   ├── base_model.py          # 基础模型类，封装通用训练 / 验证 / 编码流程（已注释）
│   │                          # 被 CIBHash.py 继承，提供 load_data / get_hparams_grid 等通用方法
│   │
│   ├── CIBHash.py             # 原始哈希模型（作为 Teacher 教师模型），支持多种骨干网络
│   │                          # 支持：vit_b_16, vit_b_32, vit_l_16, vit_h_14, mobilenet_v2
│   │                          # 通过对比学习目标将图像映射到二进制哈希码（encode_length 位）
│   │
│   ├── distill_base_model.py  # 蒸馏训练的基础模型类
│   │                          # 加载预训练教师模型，支持多种蒸馏损失（RKD / PKT / SP / DR / NST / CRD 等）
│   │
│   ├── distill_CIBHash.py     # 经典知识蒸馏哈希模型
│   │                          # 支持 efficientnet_b0 ~ b7 作为学生模型
│   │                          # 集成 CRDLoss / CRCDLoss / PACKDConLoss 等多种蒸馏损失
│   │
│   ├── ours_distill_base_model.py  # 我们方法（BRCD）的基础模型类
│   │                                # 包含蒸馏训练的完整逻辑：特征对齐 + 聚类策略 + 位掩码机制
│   │
│   ├── ours_distill_CIBHash.py      # 我们方法的哈希模型（作为 Student 学生模型）
│   │                                # 支持 mobilenet_v2 / resnet18 / resnet34 作为学生骨干网络
│   │                                # 实现了 BRCD 论文中的核心蒸馏损失
│   │
│   ├── crd/                      # 经典对比蒸馏（Contrastive Representation Distillation）模块
│   │   ├── __init__.py           # 包标识文件
│   │   ├── criterion.py         # CRDLoss：基于对比学习的蒸馏损失
│   │   │                        # 使用 teacher 特征作为锚点，从 student 侧选择正负样本进行对比
│   │   └── memory.py             # ContrastMemory：存储大量负样本的记忆库
│   │                            # 使用 Alias Method 实现高效采样，维持师生特征的缓冲区
│   │
│   ├── crcd/                     # 对比正则化蒸馏（Contrastive Regularized Distillation）模块
│   │   ├── __init__.py           # 包标识文件
│   │   ├── criterion.py         # CRCDLoss：在 CRD 基础上加入正则化约束的蒸馏损失
│   │   └── memory.py             # ContrastMemory_queue：基于队列的对比记忆库（支持动量更新）
│   │
│   └── packd/                    # Packed Asymmetric Convolution Distillation 模块
│       ├── packd.py             # PACKDConLoss：打包卷积蒸馏的对比损失实现
│       │                        # 结合 Sinkhorn 距离度量特征分布差异
│       ├── memory.py             # SimpleMemory / DequeMemory：简单记忆库和双端队列记忆库
│       │                        # 存储全量数据集特征用于负样本采样
│       └── layers.py            # SinkhornDistance：Sinkhorn 距离层，用于最优传输计算
│
├── utils/                       # 工具函数库
│   │
│   ├── __init__.py              # Python 包标识文件（使 utils 目录可被 import）
│   │
│   ├── data.py                  # 数据加载与增强核心模块
│   │                            # Data 类：管理弱增强 / 强增强（SimCLR 风格的数据增强）
│   │                            # LabeledData 类：PyTorch Dataset 实现，支持 SCID / CIFAR-10 / 自定义数据集
│   │                            # 包含图像预处理、DataLoader 构建、batch 迭代
│   │
│   ├── evaluation.py            # 评估指标计算
│   │                            # compress / distill_compress / ours_compress：生成数据库和查询的哈希码
│   │                            # calculate_top_map：计算 mAP（mean Average Precision）评估检索精度
│   │                            # calculate_hamming：计算汉明距离进行最近邻搜索
│   │                            # calculate_perceptual_metrics：感知质量评估
│   │
│   ├── logger.py               # 日志管理工具
│   │                            # Logger 类：同时输出到文件和控制台，支持性能追踪和最优模型记录
│   │
│   ├── gaussian_blur.py        # 高斯模糊图像增强算子
│   │                            # 实现 SimCLR 论文中的高斯模糊增强，用于强数据增强视图
│   │
│   └── make_dataset_index.py   # 数据集索引构建工具
│                                # 生成数据集的 train / test / database 划分索引
│
├── data/                        # 数据集存放目录
│   │
│   ├── SCID/                   # SCID 屏幕内容图像数据集（屏幕截图检索任务）
│   │   ├── train.txt           # 训练集图像路径列表（每行：路径 + 标签ID）
│   │   ├── test.txt            # 测试集图像路径列表
│   │   ├── database.txt        # 数据库集图像路径列表（用于建立检索索引）
│   │   ├── gnd_SCID.json       # 地面真值标签（Query 与数据库中相似图像的对应关系）
│   │   ├── gnd_SCID.pkl        # 同上，pickle 二进制格式，加速加载
│   │   ├── image_attack.py      # 对原始 SCID 图像施加多种攻击，生成数据库
│   │   │                        # 支持：JPEG压缩 / 裁剪 / 模糊 / 噪声 / 亮度 / 对比度 / 混合攻击
│   │   └── attack_images/       # 经攻击处理后的图像（模拟真实场景中的图像退化）
│   │
│   ├── MyScreenDataset/        # 用户自定义数据集（自定义屏幕截图数据集）
│   │   ├── train.txt           # 训练集图像路径列表
│   │   ├── test.txt            # 测试集图像路径列表
│   │   ├── database.txt        # 数据库集图像路径列表
│   │   ├── gnd.json            # 地面真值标签（JSON 格式）
│   │   ├── gnd.pkl             # 地面真值标签（pickle 格式）
│   │   ├── image_attack.py      # 同上，对自定义图像施加攻击生成测试数据
│   │   └── images/             # 原始图像存放目录
│   │
│   └── cifar10/               # CIFAR-10 数据集（自动下载）
│       └── cifar-10-batches-py/  # PyTorch 格式的 CIFAR-10 数据
│
├── scripts/                    # 一键运行脚本
│   │
│   ├── run.sh                  # 训练教师模型
│   │                          # 默认使用 ViT-B/16 在 CIFAR-10 上训练 100 epochs
│   │                          # 内含多种数据集和骨干网络的训练命令模板
│   │
│   └── ours_distill_run.sh     # 训练学生模型（BRCD 知识蒸馏）
│                                # 默认使用 MobileNet V2（学生）蒸馏自 ViT-B/16（教师）
│                                # 内含多种数据集的训练命令模板
│
├── templates/                   # Flask Web 前端模板
│   │
│   └── index.html              # 图像检索 Web 界面
│                                # 支持上传查询图片、选择 Teacher/Student 模型、显示检索结果
│
├── logs/                       # 训练日志输出目录
│
├── checkpoints/               # 模型权重保存目录
│   │                          # 命名格式：{数据集}_{学生模型}_{教师模型}__{trail}_{bit}_teacher.pt
│   │                          # 例如：scid_mobilenet_v2_vit_b_16__1_bit64_teacher.pt
│   │
│   ├── scid_vit_b_16_bit64_teacher.pt           # SCID 数据集上训练的 ViT-B/16 教师模型（64位哈希码）
│   ├── scid_mobilenet_v2_bit64_teacher.pt       # SCID 数据集上训练的 MobileNet V2 教师模型（64位哈希码）
│   ├── ours_scid_mobilenet_v2_vit_b_16__1_bit_64.pt  # BRCD 早期版本学生模型（MobileNetV2 蒸馏自 ViT-B/16）
│   ├── ours_distill_scid_mobilenet_v2_vit_b_16__1_bit_64.pt  # BRCD 蒸馏学生模型（MobileNetV2 蒸馏自 ViT-B/16）
│   └── myscreendataset_vit_b_16_bit64_teacher.pt               # 自定义数据集（MyScreenDataset）上训练的 ViT-B/16 教师模型
│
├── image/                      # 项目图片资源
│   │
│   └── framework_7.png         # 论文框架图，显示 BRCD 方法的整体流程
│
├── .idea/                      # PyCharm IDE 配置（可忽略）
│
└── README.md                   # 项目说明文档
```

---

## 目录文件速查表

| 文件 / 目录 | 类型 | 用途说明 |
|------------|------|---------|
| `app.py` | 脚本 | Flask Web 服务，提供图像检索 API 和前端页面 |
| `main.py` | 脚本 | 训练教师模型（ViT 等大型网络）|
| `ours_distill_main.py` | 脚本 | 训练学生模型（BRCD 知识蒸馏，将大模型知识迁移到小模型）|
| `create_txt_files.py` | 脚本 | 从 gnd.json 自动生成 train/test/database.txt（数据集预处理）|
| `test_logging.py` | 脚本 | 测试日志模块是否正常工作 |
| `training.log` | 日志 | 训练过程日志输出文件 |
| `model/__init__.py` | 包文件 | 标识 model 为 Python 包 |
| `model/base_model.py` | 模型 | 教师模型基类，封装通用训练流程 |
| `model/CIBHash.py` | 模型 | 教师哈希模型，支持 ViT / MobileNet 等多种骨干网络 |
| `model/distill_base_model.py` | 模型 | 蒸馏训练基类，加载预训练教师模型 |
| `model/distill_CIBHash.py` | 模型 | 经典蒸馏哈希模型（支持 EfficientNet 学生）|
| `model/ours_distill_base_model.py` | 模型 | BRCD 方法的蒸馏基类（核心训练逻辑）|
| `model/ours_distill_CIBHash.py` | 模型 | BRCD 学生哈希模型（支持 MobileNet/ResNet 学生）|
| `model/crd/` | 模块 | 经典对比蒸馏（CRD）：基于记忆库的对比学习蒸馏 |
| `model/crcd/` | 模块 | 对比正则化蒸馏（CRCD）：带队列正则化的对比蒸馏 |
| `model/packd/` | 模块 | 打包蒸馏（PACKD）：Sinkhorn 最优传输蒸馏 |
| `utils/__init__.py` | 包文件 | 标识 utils 为 Python 包 |
| `utils/data.py` | 工具 | 数据加载、图像增强（弱增强 + 强增强）、DataLoader 构建 |
| `utils/evaluation.py` | 工具 | 哈希码生成（compress）、mAP 计算、汉明距离搜索 |
| `utils/logger.py` | 工具 | 日志管理：同时写文件 + 输出控制台 |
| `utils/gaussian_blur.py` | 工具 | 高斯模糊图像增强（SimCLR 风格）|
| `utils/make_dataset_index.py` | 工具 | 数据集索引文件生成 |
| `data/SCID/` | 数据集 | SCID 屏幕内容图像数据集（含攻击图像）|
| `data/MyScreenDataset/` | 数据集 | 自定义屏幕截图数据集 |
| `data/cifar10/` | 数据集 | CIFAR-10 图像分类数据集 |
| `scripts/run.sh` | 脚本 | 一键训练教师模型 |
| `scripts/ours_distill_run.sh` | 脚本 | 一键训练学生模型（BRCD 蒸馏）|
| `templates/index.html` | 前端 | Flask 图像检索 Web 界面 |
| `logs/` | 目录 | 训练日志存放目录 |
| `checkpoints/` | 目录 | 模型权重（.pt 文件）保存目录 |
| `image/framework_7.png` | 图片 | 论文框架图 |

---

## 知识蒸馏流程说明

本项目的核心是 **BRCD（Bit-mask Robust Contrastive Knowledge Distillation）**，完整流程如下：

```
第一步：训练教师模型（Teacher）
    main.py → model/CIBHash.py → model/base_model.py
    骨干网络：ViT-B/16（大型，精度高但推理慢）
    输出：checkpoints/*.teacher.pt

第二步：知识蒸馏（BRCD）
    ours_distill_main.py → model/ours_distill_CIBHash.py → model/ours_distill_base_model.py
    教师（固定）：ViT-B/16（从第一步加载）
    学生（可训练）：MobileNet V2 / ResNet18 / ResNet34（轻量，推理快）
    蒸馏损失：对比学习对齐 + 聚类策略 + 位掩码机制
    输出：checkpoints/ours_*.pt

第三步：图像检索服务（Web）
    app.py → 加载 checkpoints 中的 teacher 和 student 模型
    提供以图搜图 API，Teacher 和 Student 模型可切换对比效果
```

---

```
