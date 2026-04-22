# python main.py cifar vit_b_16 --train --dataset my_screen_dataset --encode_length 64 --cuda --device 0 --trail 1 --epochs 5 --lr 0.0005 --num_bad_epochs 3


python main.py cifar vit_b_16 --train --dataset cifar10 --encode_length 64 --cuda --device 0 --trail 1 --epochs 200 --lr 0.0005 --num_bad_epochs 3
原始


python main.py cifar vit_b_16 --train --dataset cifar10 --encode_length 64 --cuda --device 0 --trail 1 --epochs 100 --lr 0.0005 --num_bad_epochs 3 --batch_size 128 --num_workers 4
我们服务器


python main.py scid vit_b_16 --train --dataset scid --encode_length 64 --cuda --device 0 --trail 1 --epochs 100 --lr 0.0005 --num_bad_epochs 10 --batch_size 128 --num_workers 4

python main.py scid mobilenet_v2 --train --dataset scid --encode_length 64 --cuda --device 0 --trail 3 --epochs 100 --lr 0.001 --num_bad_epochs 10 --batch_size 128 --num_workers 4

python main.py myscreendataset vit_b_16 --train --dataset my_screen_dataset --encode_length 64 --cuda --device 0 --trail 1 --epochs 100 --lr 0.0005 --num_bad_epochs 10 --batch_size 128 --num_workers 4

python main.py myscreendataset vit_b_16 --train --dataset my_screen_dataset --encode_length 64 --cuda --device 0 --trail 1 --epochs 200 --lr 0.0005 --num_bad_epochs 10 --batch_size 256 --num_workers 4
#这段代码是一个用于运行深度学习模型训练的shell脚本命令，让我逐部分解释：
#
### 命令结构分析
#
#- `python main.py`: 使用Python解释器执行[main.py](file://D:\省级大创\BRCD-main\BRCD-main\ours_distill_main.py)脚本文件
#
### 参数说明
#
#- `cifar`: 第一个位置参数，可能指定训练模式或数据处理方式
#- `vit_b_16`: 第二个位置参数，指定使用的模型架构为ViT-B/16（Vision Transformer）
#- `--train`: 启用训练模式
#- `--dataset cifar10`: 指定使用的数据集为CIFAR-10
#- `--encode_length 64`: 设置编码长度为64位（通常指哈希码长度）
#- `--cuda`: 启用CUDA支持（使用GPU加速计算）
#- `--device 4`: 指定使用第4号GPU设备
#- `--trail 1`: 设置实验追踪编号为1
#- `--epochs 200`: 设置训练轮数为200轮
#- `--lr 0.0005`: 设置学习率为0.0005
#- `--num_bad_epochs 3`: 设置容忍的无效训练轮数为3轮（可能用于早停机制）
#
#这个命令主要用于在CIFAR-10数据集上训练一个基于ViT-B/16架构的模型，进行200轮训练，使用指定的学习率和GPU设备。

# python main.py cifar vit_b_16 \
#   --train \                  # 启用训练模式
#   --dataset cifar10 \        # 指定使用的数据集为 CIFAR-10
#   --encode_length 64 \       # 设置编码长度为 64
#   --cuda \                   # 启用 CUDA 支持（使用 GPU）
#   --device 0 \               # 指定使用的 GPU 设备编号为 0
#   --trail 1 \                # 设置实验轨迹（trail）编号为 1
#   --epochs 100 \             # 设置训练轮数为 100 轮
#   --lr 0.0005 \              # 设置学习率为 0.0005
#   --num_bad_epochs 3 \       # 设置容忍无效训练 epoch 数量为 3（早停机制相关）
#   --batch_size 128 \         # 设置每个批次的样本数量为 128
#   --num_workers 4            # 设置数据加载时使用的子进程数为 4
