# ============ model/crcd/criterion_paddle.py - PaddlePaddle 版本 ============
"""
CRCD (Contrastive Representation Conditional Distillation) 损失 - PaddlePaddle 版本
"""

import paddle
import paddle.nn as nn
import math
import sys
sys.path.insert(0, '.')

eps = 1e-7


class CRCDLoss(nn.Layer):
    """CRCD Loss function - PaddlePaddle 版本"""

    def __init__(self, opt):
        super().__init__()
        self.emd_fc_type = opt.embed_type
        if self.emd_fc_type == "nofc":
            assert opt.s_dim == opt.t_dim
            opt.feat_dim = opt.s_dim

        self.embed_s = Embed(opt.s_dim, opt.feat_dim, self.emd_fc_type)
        self.embed_t = Embed(opt.t_dim, opt.feat_dim, self.emd_fc_type)

        from model.crd.criterion_paddle import ContrastMemory
        self.contrast = ContrastMemory(opt.feat_dim, opt.n_data, opt.nce_k, opt.nce_t, opt.nce_m)
        self.criterion_t = ContrastLoss(opt.n_data)
        self.criterion_s = ContrastLoss(opt.n_data)

    def forward(self, f_s, f_t, idx, contrast_idx=None):
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)
        s_loss = self.criterion_s(out_s)
        t_loss = self.criterion_t(out_t)
        loss = s_loss + t_loss
        return loss


class ContrastLoss(nn.Layer):
    """对比损失 - PaddlePaddle 版本"""

    def __init__(self, n_data):
        super().__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.shape[0]
        m = x.shape[1] - 1

        Pn = 1 / float(self.n_data)

        P_pos = paddle.index_select(x, 1, paddle.to_tensor([0]))
        log_D1 = paddle.log(P_pos / (P_pos + m * Pn + eps) + eps)

        P_neg = x[:, 1:]
        P_neg_clone = P_neg.clone()
        P_neg_clone = paddle.full_like(P_neg_clone, m * Pn)
        log_D0 = paddle.log(P_neg_clone / (P_neg + m * Pn + eps) + eps)

        loss = - (paddle.sum(log_D1) + paddle.sum(paddle.reshape(log_D0, [-1, 1]))) / bsz
        return loss[0]


class Embed(nn.Layer):
    """Embedding 模块 - PaddlePaddle 版本"""

    def __init__(self, dim_in=1024, dim_out=128, emd_fc_type='linear'):
        super().__init__()
        if emd_fc_type == "linear":
            self.linear = nn.Linear(dim_in, dim_out)
        elif emd_fc_type == "nonlinear":
            self.linear = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(),
                nn.Linear(dim_in, dim_out),
                nn.ReLU(),
                nn.Linear(dim_out, dim_out)
            )
        elif emd_fc_type == "nofc":
            self.linear = nn.Identity()
        else:
            raise NotImplementedError(emd_fc_type)

        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.reshape([x.shape[0], -1])
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Layer):
    """归一化层 - PaddlePaddle 版本"""

    def __init__(self, power=2):
        super().__init__()
        self.power = power

    def forward(self, x):
        norm = paddle.pow(paddle.sum(paddle.pow(x, self.power), axis=1, keepdim=True), 1. / self.power)
        out = x / (norm + 1e-12)
        return out
