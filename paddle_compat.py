"""
PaddlePaddle 兼容层 - PyTorch API 映射
为 BRCD 项目提供 torch -> paddle 的 API 兼容支持
"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.vision as vision_transforms
from paddle.vision import transforms as T
from paddle.io import Dataset, DataLoader
from paddle import inference
import numpy as np
import math

# ============ 核心兼容性 ============

# 将 paddle 别名为 torch，方便代码复用
torch = paddle
torchvision = vision_transforms

# ============ 自定义层/函数 ============

class CosineSimilarity(nn.Layer):
    """Paddle 实现 nn.CosineSimilarity"""
    def __init__(self, dim=1, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1, x2):
        x1_norm = paddle.sqrt(paddle.sum(x1 * x1, axis=self.dim, keepdim=True) + self.eps)
        x2_norm = paddle.sqrt(paddle.sum(x2 * x2, axis=self.dim, keepdim=True) + self.eps)
        x1 = x1 / (x1_norm + self.eps)
        x2 = x2 / (x2_norm + self.eps)
        if self.dim == 1:
            return paddle.sum(x1 * x2, axis=self.dim)
        elif self.dim == 2:
            return paddle.sum(x1 * x2, axis=self.dim)
        return x1 * x2


class Normalize(nn.Layer):
    """L2 Normalization Layer"""
    def __init__(self, power=2):
        super().__init__()
        self.power = power

    def forward(self, x):
        norm = paddle.pow(paddle.sum(paddle.pow(x, self.power), axis=1, keepdim=True), 1.0 / self.power)
        out = x / (norm + 1e-12)
        return out


class GaussianBlur(vision_transforms.transforms.BaseTransform):
    """高斯模糊 - Paddle 实现 (替代 torchvision.transforms.GaussianBlur)"""
    def __init__(self, kernel_size=5, sigma=(0.1, 2.0), keys=None):
        super().__init__(keys)
        self.kernel_size = kernel_size
        self.sigma = sigma

    def _apply_image(self, img):
        import cv2
        img = np.array(img)
        sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        if np.random.random() < 0.5:
            img = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), sigma)
        return img


class Identity(nn.Layer):
    """恒等映射层"""
    def forward(self, x):
        return x


# ============ Tensor 操作兼容 ============

def index_copy(x, index, source):
    """paddle.index_copy 等效实现"""
    # 将 source 按 index 位置写入 x
    for i, idx in enumerate(index):
        x[int(idx)] = source[i]
    return x


def fill_diagonal(a, value):
    """torch.fill_diagonal 等效实现"""
    a_np = a.numpy()
    np.fill_diagonal(a_np, value)
    return paddle.to_tensor(a_np)


def mask_correlated_samples_paddle(batch_size, N):
    """生成对比学习中排除正样本对的 mask (Paddle 版本)"""
    mask = paddle.ones([N, N], dtype='bool')
    mask = paddle.nn.functional.fill_diagonal_(mask, False)
    for i in range(batch_size):
        mask[i, batch_size + i] = False
        mask[batch_size + i, i] = False
    return mask


def diag_paddle(tensor, diagonal=0):
    """paddle.diag 等效实现"""
    if diagonal == 0:
        result = paddle.diag(tensor)
    elif diagonal > 0:
        result = paddle.diag(tensor, offset=diagonal)
    else:
        result = paddle.diag(tensor, offset=diagonal)
    return result


def scatter_nd_paddle(tensor, index, updates):
    """paddle.scatter_nd 等效 (用于 index_copy)"""
    # 对于 1D index_copy：用循环实现
    for i in range(index.shape[0]):
        tensor[index[i]] = updates[i]
    return tensor


# ============ Optimizer 兼容 ============

class OptimizerCompat:
    """Optimizer 兼容类，将 zero_grad 映射为 clear_grad"""
    @staticmethod
    def compat_optimizer(opt):
        opt.zero_grad = opt.clear_grad
        return opt


# ============ Model 训练/推理模式兼容 ============

def set_train_mode(model):
    """将模型设为训练模式"""
    model.train()

def set_eval_mode(model):
    """将模型设为评估模式"""
    model.eval()

# ============ save/load 兼容 ============

def save_model(model, path, use_tensor=False):
    """兼容 torch.save"""
    paddle.save(model.state_dict(), path)


def load_model(model, path, device='cpu'):
    """兼容 torch.load"""
    state_dict = paddle.load(path)
    model.set_state_dict(state_dict)
    return model


# ============ 预训练模型映射 ============

# torchvision.models -> paddle.vision.models 名称映射
TORCHVISION_TO_PADDLE_MODELS = {
    'efficientnet_b0': ('paddle.vision.models.efficientnet_b0', {}),
    'efficientnet_b1': ('paddle.vision.models.efficientnet_b1', {}),
    'efficientnet_b2': ('paddle.vision.models.efficientnet_b2', {}),
    'efficientnet_b3': ('paddle.vision.models.efficientnet_b3', {}),
    'efficientnet_b4': ('paddle.vision.models.efficientnet_b4', {}),
    'efficientnet_b5': ('paddle.vision.models.efficientnet_b5', {}),
    'efficientnet_b6': ('paddle.vision.models.efficientnet_b6', {}),
    'efficientnet_b7': ('paddle.vision.models.efficientnet_b7', {}),
    'resnet18': ('paddle.vision.models.resnet18', {}),
    'resnet34': ('paddle.vision.models.resnet34', {}),
    'resnet50': ('paddle.vision.models.resnet50', {}),
    'resnet101': ('paddle.vision.models.resnet101', {}),
    'resnet152': ('paddle.vision.models.resnet152', {}),
    'mobilenet_v2': ('paddle.vision.models.mobilenet_v2', {}),
}


def get_pretrained_model(model_name, pretrained=True):
    """获取预训练模型 (兼容 torchvision.models 调用方式)"""
    if model_name == 'efficientnet_b0':
        return vision_transforms.models.efficientnet_b0(pretrained=pretrained)
    elif model_name == 'efficientnet_b1':
        return vision_transforms.models.efficientnet_b1(pretrained=pretrained)
    elif model_name == 'efficientnet_b2':
        return vision_transforms.models.efficientnet_b2(pretrained=pretrained)
    elif model_name == 'efficientnet_b3':
        return vision_transforms.models.efficientnet_b3(pretrained=pretrained)
    elif model_name == 'efficientnet_b4':
        return vision_transforms.models.efficientnet_b4(pretrained=pretrained)
    elif model_name == 'efficientnet_b5':
        return vision_transforms.models.efficientnet_b5(pretrained=pretrained)
    elif model_name == 'efficientnet_b6':
        return vision_transforms.models.efficientnet_b6(pretrained=pretrained)
    elif model_name == 'efficientnet_b7':
        return vision_transforms.models.efficientnet_b7(pretrained=pretrained)
    elif model_name == 'resnet18':
        return vision_transforms.models.resnet18(pretrained=pretrained)
    elif model_name == 'resnet34':
        return vision_transforms.models.resnet34(pretrained=pretrained)
    elif model_name == 'resnet50':
        return vision_transforms.models.resnet50(pretrained=pretrained)
    elif model_name == 'resnet101':
        return vision_transforms.models.resnet101(pretrained=pretrained)
    elif model_name == 'resnet152':
        return vision_transforms.models.resnet152(pretrained=pretrained)
    elif model_name == 'mobilenet_v2':
        return vision_transforms.models.mobilenet_v2(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ============ 特殊 API ============

def detach(tensor):
    """detach 兼容"""
    return tensor.detach()


def item(tensor):
    """item() 兼容"""
    return tensor.item()


def numpy(tensor):
    """numpy() 兼容"""
    return tensor.numpy()


def to_tensor(data, dtype=None):
    """paddle.to_tensor 兼容"""
    return paddle.to_tensor(data, dtype=dtype)


def randn(*shape):
    """paddle.randn 兼容"""
    return paddle.randn(shape)


def zeros(*shape, **kwargs):
    """paddle.zeros 兼容"""
    if 'dtype' in kwargs:
        return paddle.zeros(shape, dtype=kwargs['dtype'])
    return paddle.zeros(shape)


def ones(*shape, **kwargs):
    """paddle.ones 兼容"""
    if 'dtype' in kwargs:
        return paddle.ones(shape, dtype=kwargs['dtype'])
    return paddle.ones(shape)


def arange(*args, **kwargs):
    """paddle.arange 兼容"""
    return paddle.arange(*args, **kwargs)


def randint(low, high=None, shape=None):
    """paddle.randint 兼容"""
    if high is None:
        high = low
        low = 0
    if shape is None:
        shape = [1]
    return paddle.randint(low, high, shape)


def eye(n, m=None):
    """paddle.eye 兼容"""
    return paddle.eye(n, m)


def bernoulli(tensor):
    """paddle.bernoulli 兼容"""
    return paddle.bernoulli(tensor)


def sort(tensor, axis=-1, descending=False):
    """paddle.sort 兼容"""
    return paddle.sort(tensor, axis=axis, descending=descending)


def argsort(tensor, axis=-1, descending=False):
    """paddle.argsort 兼容"""
    return paddle.argsort(tensor, axis=axis, descending=descending)


def nonzero(tensor):
    """paddle.nonzero 兼容"""
    return paddle.nonzero(tensor)


def where(condition, x, y):
    """paddle.where 兼容"""
    return paddle.where(condition, x, y)


# ============ AliasMethod for CRD ============

class AliasMethod:
    """Alias Method 高效采样 - Paddle 版本"""
    def __init__(self, probs):
        if paddle.sum(probs) > 1:
            probs = probs / paddle.sum(probs)
        K = probs.shape[0]
        self.prob = paddle.zeros([K])
        self.alias = paddle.zeros([K], dtype='int64')

        probs_np = probs.numpy()
        smaller = []
        larger = []
        for kk in range(K):
            self.prob[kk] = K * float(probs_np[kk])
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()
            self.alias[small] = large
            self.prob[large] = float(self.prob[large]) - 1.0 + float(self.prob[small])
            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller + larger:
            self.prob[last_one] = 1.0

    def draw(self, N):
        """Draw N samples from multinomial"""
        K = self.alias.shape[0]
        kk = paddle.randint(0, K, [N], dtype='int64')
        prob = self.prob[kk]
        alias = self.alias[kk]
        b = bernoulli(prob)
        oq = kk * b.astype('int64')
        oj = alias * ((1 - b.astype('int64')))
        return oq + oj


# ============ Device 兼容 ============

def to_device(tensor, device):
    """tensor.to(device) 兼容"""
    if device == 'cpu':
        return tensor.cpu()
    else:
        return tensor.cuda()


# ============ 兼容性替换函数 ============

def replace_module_device(model, device='cuda'):
    """将模型移动到指定设备"""
    model.to(device)
    return model


def manual_seed(seed):
    """设置随机种子"""
    paddle.seed(seed)
    np.random.seed(seed)


def cuda_manual_seed_all(seed):
    """CUDA 多卡随机种子"""
    paddle.seed(seed)


# ============ DataLoader workers ============

def set_num_workers(num_workers):
    """DataLoader num_workers 兼容"""
    return num_workers


# ============ 类型转换 ============

def long(tensor):
    """tensor.long() 兼容"""
    return tensor.astype('int64')


def float(tensor):
    """tensor.float() 兼容"""
    return tensor.astype('float32')


def int(tensor):
    """tensor.int() 兼容"""
    return tensor.astype('int32')


def bool(tensor):
    """tensor.bool() 兼容"""
    return tensor.astype('bool')


# ============ 常用 tensor 方法兼容 ============

class Tensor:
    """Tensor 兼容方法集合"""
    @staticmethod
    def clone(tensor):
        return tensor.clone()

    @staticmethod
    def detach(tensor):
        return tensor.detach()

    @staticmethod
    def numpy(tensor):
        return tensor.numpy()

    @staticmethod
    def item(tensor):
        return tensor.item()

    @staticmethod
    def to(tensor, device):
        if str(device).startswith('cuda'):
            return tensor.cuda()
        return tensor.cpu()

    @staticmethod
    def reshape(tensor, shape):
        return paddle.reshape(tensor, shape)

    @staticmethod
    def view(tensor, shape):
        return paddle.reshape(tensor, shape)

    @staticmethod
    def expand(tensor, shape):
        return paddle.tile(tensor, shape)

    @staticmethod
    def squeeze(tensor, axis=None):
        return paddle.squeeze(tensor, axis=axis)

    @staticmethod
    def unsqueeze(tensor, axis):
        return paddle.unsqueeze(tensor, axis)

    @staticmethod
    def transpose(tensor, axis1, axis2):
        return paddle.transpose(tensor, [i for i in range(tensor.ndim) if i not in [axis1, axis2]] + [axis2, axis1])

    @staticmethod
    def permute(tensor, order):
        return paddle.transpose(tensor, order)

    @staticmethod
    def clone(tensor):
        return tensor.clone()

    @staticmethod
    def contiguous(tensor):
        return tensor


# ============ 初始化工具 ============

def xavier_uniform_(tensor):
    """Xavier 均匀初始化"""
    import math
    fan_in = tensor.shape[0] if tensor.ndim == 1 else tensor.shape[1] if tensor.ndim == 2 else math.prod(tensor.shape)
    fan_out = tensor.shape[1] if tensor.ndim == 2 else math.prod(tensor.shape) // fan_in
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    tensor.set_value(paddle.uniform(tensor.shape, min=-limit, max=limit))
    return tensor


# ============ 打印信息 ============

def print_model_info(model):
    """打印模型参数量信息"""
    total_params = sum(p.numel().item() for p in model.parameters() if p.stop_gradient == False)
    print(f"Total trainable params: {total_params}")
