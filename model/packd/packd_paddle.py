# ============ model/packd/packd_paddle.py - PaddlePaddle 版本 ============
"""
PACKD (Prototype-wise Abstraction Contrastive Knowledge Distillation) 损失 - PaddlePaddle 版本
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import math

eps = 1e-7


class SimpleMemory(nn.Layer):
    """简单记忆库 - PaddlePaddle 版本"""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        stdv = 1. / math.sqrt(opt.feat_dim / 3)
        self.memory_bank = self.create_parameter(
            shape=[opt.n_data, opt.feat_dim],
            default_initializer=paddle.nn.initializer.Uniform(-stdv, stdv)
        )
        mem_norm = paddle.sqrt(paddle.sum(paddle.pow(self.memory_bank, 2), axis=1, keepdim=True) + 1e-12)
        self.memory_bank = self.memory_bank / (mem_norm + 1e-12)
        self.memory_change_num = 0

    def forward(self, feature, y, idx=None, update=False):
        momentum = self.opt.nce_m
        batchSize = feature.shape[0]

        with paddle.no_grad():
            weight_mem = paddle.index_select(self.memory_bank, 0, y.reshape([-1]))
            if update:
                if self.memory_change_num // 782 == 0:
                    momentum = 0
                if self.memory_change_num % 782 == 0 or self.memory_change_num % 100 == 0:
                    memory_change_diff = paddle.sum(weight_mem * feature).mean().numpy()
                    print(f'memory_change_diff: epoch {self.memory_change_num // 782}/{self.memory_change_num % 782}, {memory_change_diff}', flush=True)

                self.memory_change_num += 1
                weight_mem = weight_mem * momentum + feature * (1 - momentum)
                mem_norm = paddle.sqrt(paddle.sum(paddle.pow(weight_mem, 2), axis=1, keepdim=True) + 1e-12)
                updated_weight = weight_mem / (mem_norm + 1e-12)

                for i in range(y.shape[0]):
                    self.memory_bank[y[i]] = updated_weight[i]
                return

        weight = paddle.index_select(self.memory_bank, 0, idx.reshape([-1])).detach()
        weight = weight.reshape([batchSize, -1, self.opt.feat_dim])
        output = paddle.bmm(weight, feature.reshape([batchSize, self.opt.feat_dim, 1]))
        return output


class PACKDConLoss(nn.Layer):
    """PACKD 对比损失 - PaddlePaddle 版本"""

    def __init__(self, opt, temperature=0.07, ss_T=0.5):
        super().__init__()
        self.use_embed = True
        if opt.feat_dim >= opt.t_dim:
            opt.feat_dim = opt.t_dim
            self.use_embed = False
        else:
            self.use_embed = True

        self.temperature = temperature
        print(f't_dim: {opt.t_dim} s_dim: {opt.s_dim} feat_dim: {opt.feat_dim} temperature: {self.temperature}')

        self.embed_s = Embed(opt.s_dim, opt.feat_dim)
        self.embed_t = Embed(opt.t_dim, opt.feat_dim, use_linear=self.use_embed)

        self.memory_t = SimpleMemory(opt)

        self.iter_num = 0
        self.n_data = opt.n_data
        self.neg_k = opt.nce_k
        self.mixup_num = max(1, opt.mixup_num + 1)
        self.distance_metric = SinkhornDistance(opt.ops_eps, 100)
        self.ss_T = ss_T
        self.opt = opt

    def forward(self, feat_s, feat_t, labels=None, mask=None, contrast_idx=None, mixup_indexes=None, require_feat=False):
        batch_size = feat_s.shape[0] // self.mixup_num
        labels, idx = labels
        embed_s, _ = self.embed_s(feat_s)
        embed_t, _ = self.embed_t(feat_t)

        nor_index = (paddle.arange(self.mixup_num * batch_size) % self.mixup_num == 0).astype('bool')
        aug_index = (paddle.arange(self.mixup_num * batch_size) % self.mixup_num != 0).astype('bool')

        embed_s_nor = embed_s[nor_index]
        embed_s_aug = embed_s[aug_index]
        embed_t_nor = embed_t[nor_index]
        embed_t_aug = embed_t[aug_index]

        if isinstance(contrast_idx, list):
            contrast_idx, pos_idx = contrast_idx
        else:
            pos_idx = None

        self.memory_t(embed_t_nor, idx, contrast_idx, update=True)

        idx_expanded = idx.unsqueeze(1).expand([batch_size, self.mixup_num]).reshape([batch_size * self.mixup_num])
        if contrast_idx is not None:
            contrast_idx = contrast_idx.unsqueeze(1).expand([batch_size, self.mixup_num, -1]).reshape([batch_size * self.mixup_num, -1])

        neg_out_s_t = self.memory_t(embed_s, idx_expanded, contrast_idx)

        criterion_packd = ContrastNCELoss(self.n_data, self.mixup_num)

        ident_x = paddle.sum(embed_s * embed_t, axis=1, keepdim=True)
        neg_x = neg_out_s_t.reshape([batch_size, -1])

        embed_s = embed_s.reshape([batch_size, self.mixup_num, -1])
        embed_t = embed_t.reshape([batch_size, self.mixup_num, -1])

        cost, pi, C = self.distance_metric(embed_s, embed_t, thresh=self.opt.ops_err_thres)
        pi = pi.detach()

        pos_x = paddle.bmm(embed_s, embed_t.transpose([0, 2, 1]))
        pos_x = pos_x * pi
        pos_x = paddle.sum(pos_x * paddle.ones([1, self.mixup_num, self.mixup_num]), axis=[-2, -1]).reshape([batch_size, -1])

        loss_packd = criterion_packd(pos_x, neg_x)

        self.iter_num += 1
        if require_feat:
            return loss_packd, embed_s.reshape([batch_size * self.mixup_num, -1])
        else:
            return loss_packd


class Embed(nn.Layer):
    """Embedding 模块 - PaddlePaddle 版本"""

    def __init__(self, dim_in=1024, dim_out=128, use_linear=True, num_classes=100):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.fc = nn.Linear(dim_out, num_classes)
        self.l2norm = Normalize(2)
        self.use_embed = use_linear

    def forward(self, x):
        x = x.reshape([x.shape[0], -1])
        if self.use_embed:
            x = self.linear(x)
        x = self.l2norm(x)
        if self.use_embed:
            logits = self.fc(x * 64)
        else:
            logits = None
        return x, logits


class Normalize(nn.Layer):
    """归一化层 - PaddlePaddle 版本"""

    def __init__(self, power=2):
        super().__init__()
        self.power = power

    def forward(self, x):
        norm = paddle.pow(paddle.sum(paddle.pow(x, self.power), axis=1, keepdim=True), 1. / self.power)
        out = x / (norm + 1e-12)
        return out


class ContrastNCELoss(nn.Layer):
    """NCE 对比损失 - PaddlePaddle 版本"""

    def __init__(self, n_data, mixup_num):
        super().__init__()
        self.n_data = n_data
        self.mixup_num = mixup_num

    def forward(self, pos_x, neg_x, pi=None):
        pos_x = paddle.exp(pos_x / 0.07)
        neg_x = paddle.exp(neg_x / 0.07)
        bsz = neg_x.shape[0]
        Ng = paddle.sum(neg_x, axis=-1, keepdim=True)
        logits = paddle.log(pos_x / (pos_x + Ng) + 1e-8)

        if pi is not None:
            logits = logits * pi
            loss = -paddle.sum(logits) / (bsz // self.mixup_num)
        else:
            logits = paddle.mean(logits, axis=-1)
            loss = -paddle.sum(logits) / bsz

        return loss


class SinkhornDistance(nn.Layer):
    """Sinkhorn 距离 - PaddlePaddle 版本"""

    def __init__(self, eps, max_iter, reduction='none'):
        super().__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y, thresh=0.1):
        C = self._cost_matrix(x, y)
        x_points = x.shape[-2]
        y_points = y.shape[-2]

        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        mu = paddle.ones([batch_size, x_points], dtype='float32') / x_points
        nu = paddle.ones([batch_size, y_points], dtype='float32') / y_points

        u = paddle.zeros_like(mu)
        v = paddle.zeros_like(nu)

        for i in range(self.max_iter):
            u1 = u
            u = self.eps * (paddle.log(mu + 1e-8) - paddle.logsumexp(self.M(C, u, v), axis=-1) + 1e-8) + u
            v = self.eps * (paddle.log(nu + 1e-8) - paddle.logsumexp(self.M(C, u, v).transpose([0, 2, 1]) + 1e-8, axis=-1) + 1e-8) + v
            err = paddle.sum(paddle.abs(u - u1)) / (batch_size * x_points)
            if err.numpy() < thresh:
                break

        U, V = u, v
        pi = paddle.exp(self.M(C, U, V))
        cost = paddle.sum(pi * C, axis=[-2, -1])

        if self.reduction == 'mean':
            cost = paddle.mean(cost)
        elif self.reduction == 'sum':
            cost = paddle.sum(cost)

        return cost, pi, C

    def M(self, C, u, v):
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = paddle.sum(paddle.abs(x_col - y_lin) ** p, axis=-1)
        return C
