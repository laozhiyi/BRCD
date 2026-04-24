# PaddlePaddle 版本使用说明

本目录包含 BRCD 项目的 PyTorch 到 PaddlePaddle 的完整迁移版本。

## 迁移文件对照表

| PyTorch 版本 | PaddlePaddle 版本 |
|-------------|------------------|
| `main.py` | `main_paddle.py` |
| `ours_distill_main.py` | `ours_distill_main_paddle.py` |
| `app.py` | `app_paddle.py` |
| `utils/data.py` | `utils/data_paddle.py` |
| `utils/evaluation.py` | `utils/evaluation_paddle.py` |
| `model/CIBHash.py` | `model/CIBHash_paddle.py` |
| `model/ours_distill_CIBHash.py` | `model/ours_distill_CIBHash_paddle.py` |
| `model/crd/criterion.py` | `model/crd/criterion_paddle.py` |
| `model/crcd/criterion.py` | `model/crcd/criterion_paddle.py` |
| `model/packd/packd.py` | `model/packd/packd_paddle.py` |

## 核心 API 对照

| PyTorch | PaddlePaddle |
|---------|--------------|
| `torch` | `paddle` |
| `torchvision` | `paddle.vision` |
| `nn.Module` | `paddle.nn.Layer` |
| `nn.Module.cuda()` | `.to(gpu)` / `.cuda()` |
| `torch.no_grad()` | `paddle.no_grad()` |
| `optimizer.zero_grad()` | `optimizer.clear_grad()` |
| `torch.save` | `paddle.save` |
| `torch.load` | `paddle.load` |
| `tensor.backward()` | `tensor.backward()` |
| `nn.CrossEntropyLoss` | `nn.CrossEntropyLoss` |
| `nn.CosineSimilarity` | `自定义 CosineSimilarity` |
| `nn.SyncBatchNorm` | 不需要（飞桨自动处理） |
| `torchvision.transforms.GaussianBlur` | `transforms.GaussianBlur` |
| `torchvision.models.*` | `paddle.vision.models.*` |
| `register_buffer` | `create_parameter + stop_gradient=False` |
| `index_copy_` | 循环赋值 |
| `fill_diagonal_` | `paddle.nn.functional.fill_diagonal_` |
| `model.train()` | `model.train()` |
| `model.eval()` | `model.eval()` |

## 安装 PaddlePaddle

### CPU 版本
```bash
pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### GPU 版本（推荐）
```bash
pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 运行训练

### 教师模型训练
```bash
python main_paddle.py --dataset scid --s_model_name efficientnet_b0 \
    --encode_length 64 --batch_size 64 --epochs 60 --lr 0.001 \
    --num_workers 0
```

### 蒸馏学生模型训练
```bash
python ours_distill_main_paddle.py --dataset scid \
    --s_model_name mobilenet_v2 --t_model_name efficientnet_b0 \
    --encode_length 64 --batch_size 64 --epochs 60 --lr 0.001 \
    --num_workers 0 --temperature 0.3 --weight 0.001 --alpha 0.5
```

### 启动 Web 检索应用
```bash
python app_paddle.py
```

## 注意事项

1. **模型保存格式**: `.pdparams` 而非 `.pt`
2. **checkpoint 目录**: `./checkpoints/` 需要提前创建
3. **logs 目录**: `./logs/` 会自动创建
4. **预训练模型**: PaddlePaddle 会在首次运行时自动下载
5. **数据路径**: 确保 SCID 数据集在 `./data/SCID/` 目录下

## 已知差异

1. **AliasMethod 采样**: CRD/CRCD 损失中的 AliasMethod 高效采样使用 numpy 替代了纯 PyTorch 实现
2. **ContrastMemory**: 使用循环索引赋值替代了 `index_copy_`
3. **Sinkhorn 距离**: 在 PACKD 中使用标准 Paddle 实现
4. **预训练权重**: PaddlePaddle 的预训练权重与 PyTorch 权重**不兼容**，需要重新训练模型

## 预训练权重不兼容说明

由于 PaddlePaddle 和 PyTorch 的预训练权重格式不同，**无法直接加载 PyTorch 训练的模型**。必须使用 PaddlePaddle 版本从头训练模型。

建议训练顺序：
1. 先训练教师模型（教师模型可以是任意 backbone）
2. 再训练学生模型（使用教师模型蒸馏）
