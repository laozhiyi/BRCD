# ============ model/crd/criterion_paddle.py - PaddlePaddle 版本 ============
"""
CRD (Contrastive Representation Distillation) 损失 - PaddlePaddle 版本
"""

import paddle
import paddle.nn as nn
import math

eps = 1e-7


class CRDLoss(nn.Layer):
    """CRD Loss function - PaddlePaddle 版本"""

    def __init__(self, opt):
        super().__init__()
        self.embed_s = Embed(opt.s_dim, opt.feat_dim)
        self.embed_t = Embed(opt.t_dim, opt.feat_dim)
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
        return loss


class Embed(nn.Layer):
    """Embedding 模块 - PaddlePaddle 版本"""

    def __init__(self, dim_in=1024, dim_out=128):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
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


class ContrastMemory(nn.Layer):
    """对比记忆库 - PaddlePaddle 版本"""

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5):
        super().__init__()
        self.nLem = outputSize
        self.unigrams = paddle.ones([self.nLem])
        self.multinomial = AliasMethod(self.unigrams)
        self.K = K

        self._K = paddle.to_tensor([K], dtype='float32')
        self._T = paddle.to_tensor([T], dtype='float32')
        self._Z_v1 = paddle.to_tensor([-1.0], dtype='float32')
        self._Z_v2 = paddle.to_tensor([-1.0], dtype='float32')
        self._momentum = paddle.to_tensor([momentum], dtype='float32')

        stdv = 1. / math.sqrt(inputSize / 3)
        self.memory_v1 = self.create_parameter(
            shape=[outputSize, inputSize],
            default_initializer=paddle.nn.initializer.Uniform(-stdv, stdv)
        )
        self.memory_v2 = self.create_parameter(
            shape=[outputSize, inputSize],
            default_initializer=paddle.nn.initializer.Uniform(-stdv, stdv)
        )

    def forward(self, v1, v2, y, idx=None):
        K = int(self._K[0].numpy())
        T = float(self._T[0].numpy())
        Z_v1 = float(self._Z_v1[0].numpy())
        Z_v2 = float(self._Z_v2[0].numpy())
        momentum = float(self._momentum[0].numpy())

        batchSize = v1.shape[0]
        outputSize = self.memory_v1.shape[0]
        inputSize = self.memory_v1.shape[1]

        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1))
            idx = idx.reshape([batchSize, -1])
            idx[:, 0] = y.flatten()

        weight_v1 = paddle.index_select(self.memory_v1, 0, idx.reshape([-1])).reshape([batchSize, K + 1, inputSize])
        out_v2 = paddle.bmm(weight_v1, v1.reshape([batchSize, inputSize, 1]))
        out_v2 = paddle.exp(out_v2 / T)

        weight_v2 = paddle.index_select(self.memory_v2, 0, idx.reshape([-1])).reshape([batchSize, K + 1, inputSize])
        out_v1 = paddle.bmm(weight_v2, v2.reshape([batchSize, inputSize, 1]))
        out_v1 = paddle.exp(out_v1 / T)

        if Z_v1 < 0:
            new_Z_v1 = paddle.mean(out_v1) * outputSize
            self._Z_v1[0] = new_Z_v1
            Z_v1 = float(new_Z_v1.numpy())
            print("normalization constant Z_v1 is set to {:.1f}".format(Z_v1))

        if Z_v2 < 0:
            new_Z_v2 = paddle.mean(out_v2) * outputSize
            self._Z_v2[0] = new_Z_v2
            Z_v2 = float(new_Z_v2.numpy())
            print("normalization constant Z_v2 is set to {:.1f}".format(Z_v2))

        out_v1 = out_v1 / Z_v1
        out_v2 = out_v2 / Z_v2

        with paddle.no_grad():
            l_pos = paddle.index_select(self.memory_v1, 0, y.reshape([-1]))
            l_pos = l_pos * momentum + v1 * (1 - momentum)
            l_norm = paddle.sqrt(paddle.sum(paddle.pow(l_pos, 2), axis=1, keepdim=True) + 1e-12)
            updated_v1 = l_pos / (l_norm + 1e-12)

            ab_pos = paddle.index_select(self.memory_v2, 0, y.reshape([-1]))
            ab_pos = ab_pos * momentum + v2 * (1 - momentum)
            ab_norm = paddle.sqrt(paddle.sum(paddle.pow(ab_pos, 2), axis=1, keepdim=True) + 1e-12)
            updated_v2 = ab_pos / (ab_norm + 1e-12)

            for i in range(y.shape[0]):
                self.memory_v1[y[i]] = updated_v1[i]
                self.memory_v2[y[i]] = updated_v2[i]

        return out_v1, out_v2


class AliasMethod:
    """Alias Method 高效采样 - PaddlePaddle 版本"""

    def __init__(self, probs):
        if paddle.sum(probs) > 1:
            probs = probs / paddle.sum(probs)

        K = probs.shape[0]
        self.prob = paddle.zeros([K], dtype='float32')
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
        K = self.alias.shape[0]
        kk = paddle.randint(0, K, [N], dtype='int64')
        prob = self.prob[kk]
        alias = self.alias[kk]
        b = paddle.bernoulli(prob)
        oq = kk * b.astype('int64')
        oj = alias * ((1 - b.astype('int64')))
        return oq + oj
